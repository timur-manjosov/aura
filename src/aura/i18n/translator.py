"""Loading and lookup of Aura's translation strings.

Aura addresses every end user in their own language (CLAUDE.md,
Internationalization), so no user-facing text may be written inline anywhere
else in the codebase. This module is the single seam through which such text is
resolved: business logic passes a key and a locale and receives a string.

Invariants this module maintains
--------------------------------
* Every supported locale is loaded and validated once, at construction. A
  missing, unreadable or malformed locale file raises `TranslationLoadError`
  then -- at bot startup -- rather than the first time a user in that locale
  happens to trigger a lookup.
* `en-US` is a mandatory fallback. A key missing from a locale resolves through
  the default locale, and a key missing there too resolves to a visible
  `[key]` marker. Lookup never raises and never returns a blank string, because
  a missing translation must be a visible defect, not a bot outage.
* Adding a language is a data change: one JSON file in `locales/`, one entry in
  `SUPPORTED_LOCALES`. Nothing in this module needs to know which languages
  exist.

Imports nothing from `aura` -- it sits at the bottom of the dependency graph so
that every layer above it, including the Discord command layer, can use it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

DEFAULT_LOCALE: Final = "en-US"

SUPPORTED_LOCALES: Final[frozenset[str]] = frozenset(
    {"en-US", "de", "es-ES", "pt-BR", "fr", "tr", "pl", "ja", "ko"}
)

_LOCALES_DIR: Final = Path(__file__).parent / "locales"


class TranslationLoadError(Exception):
    """A locale file is missing, unreadable, or malformed.

    Notes
    -----
    Raised eagerly at `Translator` construction time -- that is, at bot startup
    -- rather than deferred to first lookup, so a broken locale file fails
    loudly instead of silently dropping that language until a user in it
    triggers a lookup.
    """


class Translator:
    """Resolves translation keys against locales loaded once at construction.

    Parameters
    ----------
    locales_dir
        Directory holding one `<locale>.json` file per supported locale.
        Defaults to the `locales/` directory shipped beside this module.
    supported_locales
        The locales to load. Every one of them must have a file in
        `locales_dir`.
    default_locale
        The locale every failed lookup falls back to. Must be a member of
        `supported_locales`.

    Raises
    ------
    ValueError
        If `default_locale` is not in `supported_locales`, which would leave
        the fallback path unable to resolve anything.
    TranslationLoadError
        If any supported locale's file is missing, unreadable, not a JSON
        object, or maps a key to something other than a string.

    Notes
    -----
    All I/O happens here, in the constructor; `t` is pure lookup over the
    already-parsed mapping and touches no disk.
    """

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
        """Read and validate every supported locale file.

        Parameters
        ----------
        locales_dir
            Directory to read `<locale>.json` from.
        supported_locales
            Locales to load; iterated in sorted order so a failure always
            reports the same file first for the same broken tree.

        Returns
        -------
        dict[str, dict[str, str]]
            One key-to-string mapping per locale, keyed by locale code.

        Raises
        ------
        TranslationLoadError
            On a missing file, an OS-level read failure, invalid JSON, a
            top-level value that is not an object, or any non-string value.

        Notes
        -----
        The value-type check is what makes `t`'s `str.format` call safe: a
        nested object or a number in a locale file would otherwise surface as
        an AttributeError deep inside an unrelated command.
        """
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
        """Resolve `key` for `locale`, substituting `kwargs` into its placeholders.

        Parameters
        ----------
        key
            Translation key, as written in the locale files.
        locale
            Discord locale code. An unsupported code is treated exactly like a
            supported one that is missing the key: it falls back.
        **kwargs
            Values for the template's named placeholders.

        Returns
        -------
        str
            The formatted string for `locale`; the default locale's string if
            `key` is missing there; `"[<key>]"` if it is missing from the
            default locale too; or the unformatted template if a placeholder
            could not be substituted.

        Notes
        -----
        Never raises. Each of the three degraded outcomes is logged at WARNING
        and returns something displayable, because the alternative -- an
        exception from a missing key -- turns a translation gap into a failed
        command. The bracketed key is deliberately conspicuous rather than
        blank, so the gap is reported rather than silently absorbed.
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
    """Return the process-wide `Translator`, constructing it on first call.

    Returns
    -------
    Translator
        The cached instance, built from the default locales directory.

    Raises
    ------
    TranslationLoadError
        On the first call only, if any locale file is unusable.

    Notes
    -----
    One instance per process because the locale files are read-only and
    identical for every caller; the cache exists so lookups do no disk I/O, not
    to make construction thread-safe. Two threads racing the first call would
    each build a Translator and one would win -- harmless here, since both are
    equivalent and the loser is discarded.
    """
    global _default_translator
    if _default_translator is None:
        _default_translator = Translator()
    return _default_translator


def t(key: str, locale: str, **kwargs: object) -> str:
    """Resolve a translation key for a locale using the process-wide `Translator`.

    Parameters
    ----------
    key
        Translation key, as written in the locale files.
    locale
        Discord locale code.
    **kwargs
        Values for the template's named placeholders.

    Returns
    -------
    str
        See `Translator.t`, whose fallback semantics this shares exactly.

    Notes
    -----
    This is the function the rest of the codebase calls; `Translator` is
    exposed for tests that need an isolated instance over their own fixture
    directory.
    """
    return get_translator().t(key, locale, **kwargs)
