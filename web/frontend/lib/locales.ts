/**
 * The loaded translation catalogues, and the `t()` the UI actually calls.
 *
 * Static imports rather than a dynamic fetch, so every locale ships in the
 * bundle and a language never arrives late or fails to arrive. Nine small
 * JSON files cost less than the loading state a dynamic import would need.
 *
 * Adding a language is: add the JSON file, add two lines here, add the code
 * to SUPPORTED_LOCALES. No other file changes -- CLAUDE.md's "a new language
 * -> add one locale file" principle, as close as an import list allows.
 */
import type { Catalogues } from "./i18n";
import { DEFAULT_LOCALE, resolveLocale, translate } from "./i18n";

import de from "./locales/de.json";
import enUS from "./locales/en-US.json";
import esES from "./locales/es-ES.json";
import fr from "./locales/fr.json";
import ja from "./locales/ja.json";
import ko from "./locales/ko.json";
import pl from "./locales/pl.json";
import ptBR from "./locales/pt-BR.json";
import tr from "./locales/tr.json";

export const catalogues: Catalogues = {
  "en-US": enUS,
  "es-ES": esES,
  "pt-BR": ptBR,
  de,
  fr,
  tr,
  pl,
  ja,
  ko,
};

export { DEFAULT_LOCALE, resolveLocale };

/** Resolve `key` for `locale`, falling back to en-US and then to `[key]`. */
export function t(
  key: string,
  locale: string,
  params: Record<string, string | number> = {},
): string {
  return translate(catalogues, key, locale, params);
}
