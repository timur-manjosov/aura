/**
 * Translation-key resolution for the web frontend.
 *
 * The same contract as the bot's `aura.i18n.Translator`, deliberately: keys
 * resolve against a per-locale catalogue, a missing key falls back to en-US,
 * a key missing there too renders as `[key]`, and nothing in this path ever
 * throws. CLAUDE.md's i18n section makes that fallback mandatory because a
 * missing translation must be visible, never an outage or a blank screen.
 *
 * Every function here is pure and takes its catalogues as an argument rather
 * than importing them. That is what lets the Python test suite drive this
 * file directly through node (see web/backend/tests/test_frontend_i18n.py)
 * without a second test framework, a bundler, or a browser.
 */

export const DEFAULT_LOCALE = "en-US";

/** The nine locales CLAUDE.md lists, in the order its table gives them. */
export const SUPPORTED_LOCALES = [
  "en-US",
  "es-ES",
  "pt-BR",
  "de",
  "fr",
  "tr",
  "pl",
  "ja",
  "ko",
] as const;

export type SupportedLocale = (typeof SUPPORTED_LOCALES)[number];
export type Catalogue = Record<string, string>;
export type Catalogues = Record<string, Catalogue>;

/**
 * Map whatever the browser reports to one of the supported locales.
 *
 * Three steps, narrowing: an exact match, then a language match ignoring the
 * region (so `de-AT` reaches `de` and `pt-PT` reaches `pt-BR` rather than
 * falling all the way to English), then the default. Discord's own locale
 * strings use the same shapes, so the later per-user override this leaves
 * room for can feed straight into here.
 */
export function resolveLocale(requested: string | null | undefined): SupportedLocale {
  if (typeof requested !== "string" || requested.trim() === "") {
    return DEFAULT_LOCALE;
  }
  const candidate = requested.trim();
  const exact = SUPPORTED_LOCALES.find(
    (locale) => locale.toLowerCase() === candidate.toLowerCase(),
  );
  if (exact) {
    return exact;
  }
  const language = candidate.split("-")[0].toLowerCase();
  const byLanguage = SUPPORTED_LOCALES.find(
    (locale) => locale.split("-")[0].toLowerCase() === language,
  );
  return byLanguage ?? DEFAULT_LOCALE;
}

/**
 * Resolve one key, substituting `{named}` placeholders.
 *
 * A placeholder with no matching parameter is left exactly as written rather
 * than rendered as `undefined`: the literal `{name}` on screen names the bug,
 * whereas `undefined` looks like missing data and gets reported as one.
 */
export function translate(
  catalogues: Catalogues,
  key: string,
  locale: string,
  params: Record<string, string | number> = {},
): string {
  const fallback = catalogues[DEFAULT_LOCALE] ?? {};
  const catalogue = catalogues[locale] ?? fallback;
  const template = catalogue[key] ?? fallback[key];
  if (typeof template !== "string") {
    return `[${key}]`;
  }
  return template.replace(/\{(\w+)\}/g, (whole, name: string) =>
    Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : whole,
  );
}
