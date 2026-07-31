"""Tests for aura.i18n: translation loading and key resolution.

Content-fix note (reports/i18n-content-fix.txt has the full writeup): until this
fix, all eight non-English locale files under src/aura/i18n/locales/ were
byte-identical copies of en-US.json. TestRealShippedLocales's original content
check only ever asserted `ping_response`, which is a deliberately untranslated
key (see NEVER_TRANSLATED_KEYS below) — so it passed throughout, and nothing
in this file distinguished "translated to something that happens to read the
same" from "never translated at all." TestTranslationContentDiffersFromEnglish
and TestPlaceholderParity close that gap: they check every locale/key pair
directly, not a sample.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from aura.i18n import DEFAULT_LOCALE, SUPPORTED_LOCALES, get_translator, t
from aura.i18n.translator import Translator, TranslationLoadError

_LOCALES_DIR = Path(__file__).resolve().parent.parent / "src" / "aura" / "i18n" / "locales"
_PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")

# Keys that are deliberately identical across every locale, and why:
#
# - ping_response ("Pong! 🏓"): paired with the "/ping" command name, which is
#   also never translated (see below). "Ping"/"Pong" is standard, untranslated
#   network-debug jargon in every one of these languages' tech communities;
#   translating only the reply while the command stays "/ping" would read as
#   inconsistent rather than more localized.
# - ping_command_name ("ping"): a Discord slash-command identifier. main.py
#   registers it via `translator.t("ping_command_name", DEFAULT_LOCALE)` —
#   always DEFAULT_LOCALE, never the invoking user's locale — so a translated
#   value here would never actually reach Discord under the current
#   registration code (no name_localizations wiring exists yet). Command
#   identifiers are also conventionally kept in English across Discord bots
#   for cross-server discoverability.
NEVER_TRANSLATED_KEYS = frozenset({"ping_response", "ping_command_name"})

# (locale, key) pairs where the correct, idiomatic translation is spelled
# identically to English — a genuine cognate, not a missed translation.
# Verified individually: French "Sources" is the standard French word for
# this UI context (same spelling as English), not a copy-paste leftover.
KNOWN_COGNATE_EXCEPTIONS = frozenset({
    ("fr", "ask_sources_label"),
})


def _load_locale_json(locale: str) -> dict[str, str]:
    return json.loads((_LOCALES_DIR / f"{locale}.json").read_text(encoding="utf-8"))


def _write_locale(directory: Path, locale: str, content: object) -> None:
    """Write a locale file. A str `content` is written verbatim (for malformed-JSON
    cases); anything else is JSON-encoded first."""
    text = content if isinstance(content, str) else json.dumps(content)
    (directory / f"{locale}.json").write_text(text, encoding="utf-8")


class TestRealShippedLocales:
    """Sanity checks against the actual files under src/aura/i18n/locales/."""

    def test_all_supported_locales_load_without_error(self) -> None:
        translator = get_translator()
        for locale in SUPPORTED_LOCALES:
            assert translator.t("ping_response", locale) == "Pong! 🏓"

    def test_module_level_convenience_function_works(self) -> None:
        assert t("ping_response", DEFAULT_LOCALE) == "Pong! 🏓"

    def test_get_translator_returns_the_same_cached_instance(self) -> None:
        assert get_translator() is get_translator()


class TestTranslationContentDiffersFromEnglish:
    """Guards against a locale file that loads fine but was never actually translated.

    This is the check that was missing: TestRealShippedLocales confirms the files
    *load*, this confirms their *content* is real translation work, key by key,
    locale by locale — not a sample.
    """

    _EN = _load_locale_json(DEFAULT_LOCALE)
    _NON_ENGLISH_LOCALES = sorted(SUPPORTED_LOCALES - {DEFAULT_LOCALE})

    @pytest.mark.parametrize("locale", _NON_ENGLISH_LOCALES)
    def test_locale_has_same_key_set_as_english(self, locale: str) -> None:
        data = _load_locale_json(locale)
        assert set(data.keys()) == set(self._EN.keys())

    @pytest.mark.parametrize("locale", _NON_ENGLISH_LOCALES)
    def test_every_translatable_key_differs_from_english(self, locale: str) -> None:
        data = _load_locale_json(locale)
        untranslated = [
            key
            for key, en_value in self._EN.items()
            if key not in NEVER_TRANSLATED_KEYS
            and (locale, key) not in KNOWN_COGNATE_EXCEPTIONS
            and data.get(key) == en_value
        ]
        assert not untranslated, (
            f"{locale}.json has keys identical to en-US.json that aren't in "
            f"NEVER_TRANSLATED_KEYS or KNOWN_COGNATE_EXCEPTIONS: {untranslated}"
        )


class TestPlaceholderParity:
    """Every {placeholder} in an English string must appear, verbatim, in every
    translation of that string — a renamed or dropped placeholder is a silent
    KeyError at format time, not at load time, so this has to be checked by content
    inspection rather than by the loader succeeding."""

    _EN = _load_locale_json(DEFAULT_LOCALE)
    _NON_ENGLISH_LOCALES = sorted(SUPPORTED_LOCALES - {DEFAULT_LOCALE})

    @pytest.mark.parametrize("locale", _NON_ENGLISH_LOCALES)
    def test_placeholders_match_english_exactly(self, locale: str) -> None:
        data = _load_locale_json(locale)
        mismatches = {}
        for key, en_value in self._EN.items():
            en_placeholders = set(_PLACEHOLDER_RE.findall(en_value))
            locale_placeholders = set(_PLACEHOLDER_RE.findall(data.get(key, "")))
            if en_placeholders != locale_placeholders:
                mismatches[key] = {
                    "expected": en_placeholders,
                    "got": locale_placeholders,
                }
        assert not mismatches, f"{locale}.json placeholder mismatches: {mismatches}"


class TestFallbackBehavior:
    @pytest.fixture
    def translator(self, tmp_path: Path) -> Translator:
        supported = frozenset({"en-US", "de"})
        _write_locale(tmp_path, "en-US", {"greeting": "Hello", "farewell": "Bye"})
        _write_locale(tmp_path, "de", {"greeting": "Hallo"})  # farewell intentionally missing
        return Translator(locales_dir=tmp_path, supported_locales=supported, default_locale="en-US")

    def test_key_present_in_requested_locale(self, translator: Translator) -> None:
        assert translator.t("greeting", "de") == "Hallo"

    def test_key_missing_in_requested_locale_falls_back_to_default(
        self, translator: Translator
    ) -> None:
        assert translator.t("farewell", "de") == "Bye"

    def test_key_missing_everywhere_returns_bracketed_key(self, translator: Translator) -> None:
        assert translator.t("does_not_exist", "de") == "[does_not_exist]"

    def test_unsupported_locale_falls_back_to_default_entirely(
        self, translator: Translator
    ) -> None:
        assert translator.t("greeting", "xx-XX") == "Hello"

    def test_empty_string_locale_falls_back_to_default(self, translator: Translator) -> None:
        assert translator.t("greeting", "") == "Hello"

    def test_empty_string_key_does_not_crash(self, translator: Translator) -> None:
        assert translator.t("", "en-US") == "[]"

    def test_nonexistent_key_in_nonexistent_locale_returns_bracketed_key(
        self, translator: Translator
    ) -> None:
        assert translator.t("totally_bogus_key", "xx-YY") == "[totally_bogus_key]"


class TestFormatting:
    @pytest.fixture
    def translator(self, tmp_path: Path) -> Translator:
        supported = frozenset({"en-US"})
        _write_locale(tmp_path, "en-US", {"welcome": "Welcome, {name}!"})
        return Translator(locales_dir=tmp_path, supported_locales=supported, default_locale="en-US")

    def test_placeholder_is_substituted(self, translator: Translator) -> None:
        assert translator.t("welcome", "en-US", name="Ada") == "Welcome, Ada!"

    def test_unicode_kwarg_is_substituted_correctly(self, translator: Translator) -> None:
        assert translator.t("welcome", "en-US", name="日本語ユーザー") == "Welcome, 日本語ユーザー!"

    def test_missing_kwarg_degrades_to_raw_template_instead_of_raising(
        self, translator: Translator
    ) -> None:
        assert translator.t("welcome", "en-US") == "Welcome, {name}!"


class TestLoadFailures:
    """Locale files must fail loudly at construction (i.e. bot startup), never silently."""

    def test_missing_locale_file_raises(self, tmp_path: Path) -> None:
        supported = frozenset({"en-US", "de"})
        _write_locale(tmp_path, "en-US", {"a": "b"})
        # "de" is deliberately never written.
        with pytest.raises(TranslationLoadError, match="de"):
            Translator(locales_dir=tmp_path, supported_locales=supported, default_locale="en-US")

    def test_invalid_json_syntax_raises(self, tmp_path: Path) -> None:
        supported = frozenset({"en-US"})
        _write_locale(tmp_path, "en-US", "{not valid json,,,")
        with pytest.raises(TranslationLoadError, match="en-US"):
            Translator(locales_dir=tmp_path, supported_locales=supported, default_locale="en-US")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        supported = frozenset({"en-US"})
        _write_locale(tmp_path, "en-US", "")
        with pytest.raises(TranslationLoadError):
            Translator(locales_dir=tmp_path, supported_locales=supported, default_locale="en-US")

    def test_json_array_instead_of_object_raises(self, tmp_path: Path) -> None:
        supported = frozenset({"en-US"})
        _write_locale(tmp_path, "en-US", ["not", "an", "object"])
        with pytest.raises(TranslationLoadError):
            Translator(locales_dir=tmp_path, supported_locales=supported, default_locale="en-US")

    def test_non_string_value_raises(self, tmp_path: Path) -> None:
        supported = frozenset({"en-US"})
        _write_locale(tmp_path, "en-US", {"ping_response": 42})
        with pytest.raises(TranslationLoadError):
            Translator(locales_dir=tmp_path, supported_locales=supported, default_locale="en-US")

    def test_nested_object_value_raises(self, tmp_path: Path) -> None:
        supported = frozenset({"en-US"})
        _write_locale(tmp_path, "en-US", {"ping_response": {"nested": "not flat"}})
        with pytest.raises(TranslationLoadError):
            Translator(locales_dir=tmp_path, supported_locales=supported, default_locale="en-US")

    def test_default_locale_not_in_supported_set_raises_immediately(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="default_locale"):
            Translator(
                locales_dir=tmp_path,
                supported_locales=frozenset({"de"}),
                default_locale="en-US",
            )


class TestConcurrentAccess:
    async def test_concurrent_lookups_are_consistent(self, tmp_path: Path) -> None:
        supported = frozenset({"en-US"})
        _write_locale(tmp_path, "en-US", {"greeting": "Hello, {name}!"})
        translator = Translator(
            locales_dir=tmp_path, supported_locales=supported, default_locale="en-US"
        )

        async def lookup(n: int) -> str:
            return await asyncio.to_thread(translator.t, "greeting", "en-US", name=str(n))

        results = await asyncio.gather(*(lookup(i) for i in range(200)))
        assert results == [f"Hello, {i}!" for i in range(200)]
