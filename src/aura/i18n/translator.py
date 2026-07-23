"""Loading and lookup of Aura's translation strings."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LOCALE = "en-US"

SUPPORTED_LOCALES: frozenset[str] = frozenset(
    {"en-US", "de", "es-ES", "pt-BR", "fr", "tr", "pl", "ja", "ko"}
)

_LOCALES_DIR = Path(__file__).parent / "locales"


class TranslationLoadError(Exception):
    """Raised when a locale file is missing, unreadable, or malformed.

    Intentionally raised eagerly at Translator construction time (i.e. at
    bot startup) rather than deferred to first lookup, so a broken locale
    file fails loudly instead of silently dropping that language the first
    time a user in that locale triggers a lookup.
    """


class Translator:
    """Resolves translation keys against a fixed set of locales loaded once at construction."""

    def __init__(
        self,
        locales_dir: Path = _LOCALES_DIR,
        supported_locales: frozenset[str] = SUPPORTED_LOCALES,
        default_locale: str = DEFAULT_LOCALE,
    ) -> None:
        if default_locale not in supported_locales:
            raise ValueError("default_locale must be a member of supported_locales")
        self._default_locale = default_locale
        self._translations = self._load_all(locales_dir, supported_locales)

    @staticmethod
    def _load_all(
        locales_dir: Path, supported_locales: frozenset[str]
    ) -> dict[str, dict[str, str]]:
        translations: dict[str, dict[str, str]] = {}
        for locale in sorted(supported_locales):
            locale_path = locales_dir / f"{locale}.json"
            if not locale_path.is_file():
                raise TranslationLoadError(f"Missing locale file: {locale_path}")

            try:
                raw_text = locale_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise TranslationLoadError(
                    f"Could not read locale file {locale_path}: {exc}"
                ) from exc

            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise TranslationLoadError(
                    f"Invalid JSON in locale file {locale_path}: {exc}"
                ) from exc

            if not isinstance(data, dict):
                raise TranslationLoadError(
                    f"Locale file {locale_path} must contain a JSON object mapping keys "
                    f"to strings, got {type(data).__name__}"
                )
            for translation_key, value in data.items():
                if not isinstance(value, str):
                    raise TranslationLoadError(
                        f"Locale file {locale_path} must map keys to string values "
                        f"(key {translation_key!r} has a {type(value).__name__} value)"
                    )

            translations[locale] = data

        return translations

    def t(self, key: str, locale: str, **kwargs: object) -> str:
        """Resolve `key` for `locale`.

        Falls back to the default locale if `key` is missing in `locale`
        (or `locale` itself is unsupported), and to a bracketed key if `key`
        is missing from the default locale too. Never raises for a missing
        key/locale/placeholder — a missing translation must be visible, not
        a bot outage.
        """
        default_map = self._translations[self._default_locale]
        locale_map = self._translations.get(locale, default_map)

        template = locale_map.get(key, default_map.get(key))
        if template is None:
            logger.warning(
                "Missing translation key %r (locale %r, fallback %r)",
                key,
                locale,
                self._default_locale,
            )
            return f"[{key}]"

        try:
            return template.format(**kwargs)
        except (KeyError, IndexError) as exc:
            logger.warning(
                "Failed to format translation key %r for locale %r: %s", key, locale, exc
            )
            return template


_default_translator: Translator | None = None


def get_translator() -> Translator:
    """Return the process-wide Translator, constructing and caching it on first call."""
    global _default_translator
    if _default_translator is None:
        _default_translator = Translator()
    return _default_translator


def t(key: str, locale: str, **kwargs: object) -> str:
    """Resolve a translation key for a locale using the process-wide Translator.

    See Translator.t for fallback semantics.
    """
    return get_translator().t(key, locale, **kwargs)
