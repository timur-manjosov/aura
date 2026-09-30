"""Every bot credential is a SecretStr until the one place it is used (F-10).

The failure this guards against is quiet in both directions. A credential typed
`str` prints through any repr of the settings; a SecretStr that is NOT
unwrapped where it is used sends "**********" to Discord, to the LLM provider
or into the internal API's expected header -- an outage, or worse, a listener
whose secret is the mask. So each consumer is driven for real and the value it
received is checked for its TYPE and its exact content, and the settings are
checked to print nothing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import aiosqlite
import pytest
from aiohttp.test_utils import TestClient, TestServer
from pydantic import SecretStr

from aura.billing import PlanGate
from aura.billing.internal_api import create_internal_api_app
from aura.config import ModelComponent, Settings
from aura.db.extraction_queue import QueuedMessage
from aura.db.models import Fact, FactStatus
from aura.db.repository import init_schema
from aura.extraction.distiller import distill_facts
from aura.extraction.supersession import judge_relationship
from aura.grounding import verify_answer_grounded
from aura.synthesis import synthesize_answer
from aura.variants_service import _audit_variants, _generate_variants

MASK = "**********"
LLM_KEY = "sk-or-plain-llm-key-value"
DISCORD_TOKEN = "plain.discord.token-value"
INTERNAL_SECRET = "plain-internal-api-secret-" + "k" * 20


def settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token=DISCORD_TOKEN,  # type: ignore[arg-type]
        llm_api_key=LLM_KEY,  # type: ignore[arg-type]
        internal_api_secret=INTERNAL_SECRET,  # type: ignore[arg-type]
        synthesis_model="openrouter/fake/model",
        variant_audit_model="openrouter/fake/auditor",
        grounding_check_model="openrouter/fake/checker",
    )


def fact() -> Fact:
    return Fact(
        id=1,
        guild_id=100000000000000001,
        channel_id=11,
        message_id=101,
        content="The server rules are pinned in #rules.",
        embedding=b"",
        status=FactStatus.ACTIVE,
        superseded_by_id=None,
        created_at=datetime.now(UTC),
    )


class ProviderUnavailableError(Exception):
    """Raised by the stub provider after it has recorded the call."""


def recording_provider(calls: list[dict[str, Any]]) -> Callable[..., Awaitable[Any]]:
    async def provider(**kwargs: Any) -> Any:
        calls.append(kwargs)
        raise ProviderUnavailableError

    return provider


async def call_synthesis() -> None:
    await synthesize_answer([fact()], "where are the rules?", "en-US", model="m")


async def call_distiller() -> None:
    created = datetime.now(UTC)
    await distill_facts(
        [
            QueuedMessage(
                channel_id=500,
                message_id=1,
                guild_id=100,
                channel_name="announcements",
                content="The event is on Saturday.",
                message_created_at=created,
                enqueued_at=created,
            )
        ],
        channel_name="announcements",
        model="m",
    )


async def call_supersession() -> None:
    await judge_relationship(predecessor="old fact", candidate="new fact", model="m")


async def call_variant_generation() -> None:
    await _generate_variants("The rules are pinned.", count=3, model="m")


async def call_variant_audit() -> None:
    await _audit_variants(canonical="The rules are pinned.", variants=["a", "b"], model="m")


async def call_grounding() -> None:
    await verify_answer_grounded(
        answer="They are pinned.", cited_facts=[fact()], settings=settings(), timeout_seconds=5
    )


LLM_CALL_SITES: dict[str, tuple[str, Callable[[], Awaitable[None]]]] = {
    "synthesis": ("aura.synthesis", call_synthesis),
    "distiller": ("aura.extraction.distiller", call_distiller),
    "supersession": ("aura.extraction.supersession", call_supersession),
    "variant-generation": ("aura.variants_service", call_variant_generation),
    "variant-audit": ("aura.variants_service", call_variant_audit),
    "grounding": ("aura.grounding", call_grounding),
}


class TestTheSettingsTypeEveryCredential:
    @pytest.mark.parametrize("field", ["discord_token", "llm_api_key", "internal_api_secret"])
    def test_each_credential_is_a_secret_str(self, field: str) -> None:
        assert isinstance(getattr(settings(), field), SecretStr)

    def test_nothing_the_settings_print_contains_a_credential(self) -> None:
        configured = settings()
        printed = "".join(
            (
                repr(configured),
                str(configured),
                str(configured.model_dump()),
                configured.model_dump_json(),
            )
        )

        for secret in (LLM_KEY, DISCORD_TOKEN, INTERNAL_SECRET):
            assert secret not in printed
        assert MASK in printed

    def test_a_blank_key_is_still_not_configured(self) -> None:
        blank = Settings(
            _env_file=None,  # type: ignore[call-arg]
            discord_token="t",  # type: ignore[arg-type]
            llm_api_key="",  # type: ignore[arg-type]
            synthesis_model="m",
        )

        assert blank.is_llm_configured(ModelComponent.SYNTHESIS) is False


class TestEveryLlmCallSiteSendsThePlainKey:
    @pytest.mark.parametrize("site", list(LLM_CALL_SITES))
    async def test_the_provider_receives_a_plain_str_that_is_the_key(
        self, site: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, call = LLM_CALL_SITES[site]
        if module != "aura.grounding":
            monkeypatch.setattr(f"{module}.load_settings", settings)
        calls: list[dict[str, Any]] = []

        with patch(f"{module}.litellm.acompletion", recording_provider(calls)):
            await call()

        assert len(calls) == 1
        assert type(calls[0]["api_key"]) is str
        assert calls[0]["api_key"] == LLM_KEY


class TestTheDiscordLoginReceivesThePlainToken:
    def test_client_run_is_given_the_token_itself(self) -> None:
        from aura import main as entry

        client = MagicMock()
        with (
            patch.object(entry, "load_settings", settings),
            patch.object(entry, "configure_logging"),
            patch.object(entry, "create_client", return_value=client),
        ):
            entry.main()

        (token,), _ = client.run.call_args
        assert type(token) is str
        assert token == DISCORD_TOKEN


class TestTheInternalApiComparesAgainstTheSecretItself:
    async def test_the_mask_is_not_a_password(self) -> None:
        """Were the SecretStr interpolated unwrapped, "Bearer **********" would be let in."""
        conn = await aiosqlite.connect(":memory:")
        await init_schema(conn)
        app = create_internal_api_app(
            conn, PlanGate.unenforced(), secret=SecretStr(INTERNAL_SECRET)
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            body = {"guild_ids": ["1000"]}

            masked = await client.post(
                "/internal/v1/guilds/plans",
                json=body,
                headers={"Authorization": f"Bearer {MASK}"},
            )
            genuine = await client.post(
                "/internal/v1/guilds/plans",
                json=body,
                headers={"Authorization": f"Bearer {INTERNAL_SECRET}"},
            )
        finally:
            await client.close()
            await conn.close()

        assert masked.status == 401
        assert genuine.status == 200
