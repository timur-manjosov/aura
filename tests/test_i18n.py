"""Tests for aura.i18n: translation loading and key resolution."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from aura.i18n import DEFAULT_LOCALE, SUPPORTED_LOCALES, get_translator, t
from aura.i18n.translator import Translator, TranslationLoadError


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
