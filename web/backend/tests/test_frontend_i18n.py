"""The frontend's locale files and its translation function, tested from pytest.

CLAUDE.md makes two i18n promises that a web frontend can break as easily as
the bot could: every user-facing string comes from a locale file, and a
missing key falls back to en-US rather than crashing or rendering blank. The
project's acceptance is automated tests, so both are checked here rather than
by reading the TypeScript.

The catalogue checks are plain JSON assertions. The behavioural checks drive
lib/i18n.ts through node, which executes TypeScript directly -- so the real
shipped function is exercised without adding a second test framework to the
repository. They skip, rather than fail, where node is absent; the catalogue
checks (the part that actually rots as languages are added) always run.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
LOCALES_DIR = FRONTEND / "lib" / "locales"
I18N_MODULE = FRONTEND / "lib" / "i18n.ts"

# CLAUDE.md's Internationalization table, verbatim. Hardcoded rather than
# derived from the directory listing: a test that reads the same directory it
# validates would happily pass after someone deleted a language.
REQUIRED_LOCALES = {"en-US", "es-ES", "pt-BR", "de", "fr", "tr", "pl", "ja", "ko"}
DEFAULT_LOCALE = "en-US"

PLACEHOLDER = re.compile(r"\{(\w+)\}")


def load_catalogue(locale: str) -> dict[str, str]:
    return json.loads((LOCALES_DIR / f"{locale}.json").read_text(encoding="utf-8"))


def node_available() -> bool:
    return shutil.which("node") is not None


def run_in_node(script: str) -> str:
    """Execute a snippet against the real lib/i18n.ts and return its stdout."""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        cwd=FRONTEND,
        timeout=60,
        # The return code is asserted below rather than raised on, so the
        # failure message carries node's stderr instead of a bare exit status.
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


class TestCatalogueFiles:
    def test_every_required_locale_has_a_file(self) -> None:
        present = {path.stem for path in LOCALES_DIR.glob("*.json")}

        assert REQUIRED_LOCALES <= present, REQUIRED_LOCALES - present

    def test_no_unexpected_locale_file_is_shipped(self) -> None:
        """A stray file would ship an unreviewed language to real users."""
        present = {path.stem for path in LOCALES_DIR.glob("*.json")}

        assert present == REQUIRED_LOCALES

    @pytest.mark.parametrize("locale", sorted(REQUIRED_LOCALES))
    def test_every_file_is_a_flat_object_of_strings(self, locale: str) -> None:
        catalogue = load_catalogue(locale)

        assert isinstance(catalogue, dict)
        assert all(isinstance(key, str) for key in catalogue)
        assert all(isinstance(value, str) for value in catalogue.values())

    @pytest.mark.parametrize("locale", sorted(REQUIRED_LOCALES))
    def test_every_locale_has_exactly_the_default_locales_keys(self, locale: str) -> None:
        """Missing keys silently fall back to English; extra keys are dead weight."""
        reference = set(load_catalogue(DEFAULT_LOCALE))
        catalogue = set(load_catalogue(locale))

        assert catalogue == reference, {
            "missing": sorted(reference - catalogue),
            "extra": sorted(catalogue - reference),
        }

    @pytest.mark.parametrize("locale", sorted(REQUIRED_LOCALES))
    def test_no_value_is_blank(self, locale: str) -> None:
        """A blank translation is worse than a missing one: it never falls back."""
        blanks = [key for key, value in load_catalogue(locale).items() if not value.strip()]

        assert blanks == []

    @pytest.mark.parametrize("locale", sorted(REQUIRED_LOCALES - {DEFAULT_LOCALE}))
    def test_placeholders_match_the_default_locale_exactly(self, locale: str) -> None:
        """A translated string that dropped {name} renders a sentence with a hole in it."""
        reference = load_catalogue(DEFAULT_LOCALE)
        catalogue = load_catalogue(locale)

        mismatched = {
            key: (sorted(PLACEHOLDER.findall(reference[key])), sorted(PLACEHOLDER.findall(value)))
            for key, value in catalogue.items()
            if sorted(PLACEHOLDER.findall(reference[key])) != sorted(PLACEHOLDER.findall(value))
        }

        assert mismatched == {}

    @pytest.mark.parametrize("locale", sorted(REQUIRED_LOCALES))
    def test_no_value_carries_a_control_character(self, locale: str) -> None:
        offenders = [
            key
            for key, value in load_catalogue(locale).items()
            if any(character != "\n" and not character.isprintable() for character in value)
        ]

        assert offenders == []

    def test_the_module_lists_exactly_the_locales_on_disk(self) -> None:
        """SUPPORTED_LOCALES and the directory must not drift apart."""
        source = I18N_MODULE.read_text(encoding="utf-8")
        block = source.split("SUPPORTED_LOCALES = [", 1)[1].split("]", 1)[0]
        declared = set(re.findall(r'"([^"]+)"', block))

        assert declared == REQUIRED_LOCALES


@pytest.mark.skipif(not node_available(), reason="node is not installed")
class TestTranslateBehaviour:
    """The fallback contract CLAUDE.md makes mandatory, exercised in the real runtime."""

    def test_a_known_key_resolves_in_its_own_locale(self) -> None:
        output = run_in_node(
            """
            import { translate } from './lib/i18n.ts';
            const catalogues = { 'en-US': { greet: 'Hello' }, de: { greet: 'Hallo' } };
            console.log(translate(catalogues, 'greet', 'de'));
            """
        )

        assert output == "Hallo"

    def test_a_key_missing_from_a_locale_falls_back_to_english(self) -> None:
        output = run_in_node(
            """
            import { translate } from './lib/i18n.ts';
            const catalogues = { 'en-US': { greet: 'Hello' }, de: {} };
            console.log(translate(catalogues, 'greet', 'de'));
            """
        )

        assert output == "Hello"

    def test_an_unsupported_locale_falls_back_to_english(self) -> None:
        output = run_in_node(
            """
            import { translate } from './lib/i18n.ts';
            const catalogues = { 'en-US': { greet: 'Hello' } };
            console.log(translate(catalogues, 'greet', 'xx-YY'));
            """
        )

        assert output == "Hello"

    def test_a_key_missing_everywhere_renders_visibly_rather_than_blank(self) -> None:
        """The bot's Translator does the same. A blank message reads as a bug in the data."""
        output = run_in_node(
            """
            import { translate } from './lib/i18n.ts';
            console.log(translate({ 'en-US': {} }, 'nope', 'de'));
            """
        )

        assert output == "[nope]"

    def test_an_empty_catalogue_set_does_not_throw(self) -> None:
        output = run_in_node(
            """
            import { translate } from './lib/i18n.ts';
            console.log(translate({}, 'anything', 'de'));
            """
        )

        assert output == "[anything]"

    def test_placeholders_are_substituted(self) -> None:
        output = run_in_node(
            """
            import { translate } from './lib/i18n.ts';
            const catalogues = { 'en-US': { hi: 'Hi {name}, you have {n}' } };
            console.log(translate(catalogues, 'hi', 'en-US', { name: 'Timur', n: 3 }));
            """
        )

        assert output == "Hi Timur, you have 3"

    def test_a_missing_parameter_leaves_the_placeholder_visible(self) -> None:
        """`undefined` on screen looks like missing data; `{name}` names the bug."""
        output = run_in_node(
            """
            import { translate } from './lib/i18n.ts';
            console.log(translate({ 'en-US': { hi: 'Hi {name}' } }, 'hi', 'en-US'));
            """
        )

        assert output == "Hi {name}"

    def test_every_real_key_resolves_in_every_real_locale(self) -> None:
        """The end-to-end contract, over the shipped catalogues rather than fixtures."""
        output = run_in_node(
            """
            import { readFileSync, readdirSync } from 'node:fs';
            import { translate } from './lib/i18n.ts';
            const dir = './lib/locales';
            const catalogues = {};
            for (const file of readdirSync(dir)) {
              catalogues[file.replace(/\\.json$/, '')] = JSON.parse(readFileSync(`${dir}/${file}`, 'utf8'));
            }
            const keys = Object.keys(catalogues['en-US']);
            const broken = [];
            for (const locale of Object.keys(catalogues)) {
              for (const key of keys) {
                const value = translate(catalogues, key, locale, { name: 'X' });
                if (!value || value === `[${key}]` || /\\{\\w+\\}/.test(value)) {
                  broken.push(`${locale}:${key}`);
                }
              }
            }
            console.log(JSON.stringify(broken));
            """
        )

        assert json.loads(output) == []


@pytest.mark.skipif(not node_available(), reason="node is not installed")
class TestResolveLocale:
    def test_an_exact_match_wins(self) -> None:
        assert self._resolve("pt-BR") == "pt-BR"

    def test_a_region_variant_falls_back_to_its_language(self) -> None:
        """de-AT must reach German, not English."""
        assert self._resolve("de-AT") == "de"

    def test_portuguese_from_portugal_reaches_the_brazilian_catalogue(self) -> None:
        assert self._resolve("pt-PT") == "pt-BR"

    def test_english_from_anywhere_reaches_the_default(self) -> None:
        assert self._resolve("en-GB") == "en-US"

    def test_matching_is_case_insensitive(self) -> None:
        assert self._resolve("DE") == "de"

    @pytest.mark.parametrize("value", ["", "   ", "zz", "klingon", "!!!", "en_US"])
    def test_an_unknown_or_malformed_value_falls_back_to_the_default(self, value: str) -> None:
        assert self._resolve(value) == "en-US"

    def test_a_missing_value_falls_back_to_the_default(self) -> None:
        output = run_in_node(
            """
            import { resolveLocale } from './lib/i18n.ts';
            console.log([resolveLocale(null), resolveLocale(undefined)].join(','));
            """
        )

        assert output == "en-US,en-US"

    @staticmethod
    def _resolve(value: str) -> str:
        payload = json.dumps(value)
        return run_in_node(
            f"""
            import {{ resolveLocale }} from './lib/i18n.ts';
            console.log(resolveLocale({payload}));
            """
        )
