"use client";

/**
 * The whole shell: sign in, see who you are, see the servers you can manage.
 *
 * One client component on purpose. Everything on this page depends on a
 * session cookie the server component tree cannot read without forwarding
 * headers, and 4b has no content worth rendering before that is known -- so
 * the page loads, asks the backend who it is talking to, and renders one of
 * three states. When 4d adds real content, that is the point to split.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  fetchCurrentUser,
  fetchManageableGuilds,
  guildIconUrl,
  guildInitial,
  logout,
  type CurrentUser,
  type ManageableGuild,
} from "@/lib/api";
import { DEFAULT_LOCALE, resolveLocale, t } from "@/lib/locales";

type LoadState =
  | { status: "loading" }
  | { status: "anonymous" }
  | { status: "signed-in"; user: CurrentUser; guilds: ManageableGuild[] }
  | { status: "error"; code: string };

export default function HomePage() {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [locale, setLocale] = useState<string>(DEFAULT_LOCALE);

  // Read the browser's language after mount, never during render: the server
  // has no navigator, and reading it during render would make the first
  // client paint disagree with the server's HTML.
  useEffect(() => {
    setLocale(resolveLocale(typeof navigator === "undefined" ? null : navigator.language));
  }, []);

  const translate = useMemo(
    () => (key: string, params?: Record<string, string | number>) => t(key, locale, params),
    [locale],
  );

  const load = useCallback(async () => {
    setState({ status: "loading" });
    const me = await fetchCurrentUser();
    if (me.kind === "error") {
      // Not being signed in is the ordinary first visit, not a failure.
      setState(
        me.code === "not_authenticated"
          ? { status: "anonymous" }
          : { status: "error", code: me.code },
      );
      return;
    }

    const guilds = await fetchManageableGuilds();
    if (guilds.kind === "error") {
      setState(
        guilds.code === "not_authenticated"
          ? { status: "anonymous" }
          : { status: "error", code: guilds.code },
      );
      return;
    }
    setState({ status: "signed-in", user: me.value, guilds: guilds.value });
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const signOut = useCallback(async () => {
    await logout();
    setState({ status: "anonymous" });
  }, []);

  return (
    <main className="page">
      <header className="masthead">
        <div>
          <h1>{translate("app_name")}</h1>
          <p className="tagline">{translate("app_tagline")}</p>
        </div>
        {state.status === "signed-in" && (
          <div className="identity">
            <span>
              {translate("signed_in_as", {
                name: state.user.global_name ?? state.user.username,
              })}
            </span>
            <button type="button" className="button secondary" onClick={() => void signOut()}>
              {translate("logout_button")}
            </button>
          </div>
        )}
      </header>

      <p className="notice">{translate("phase_notice")}</p>

      {state.status === "loading" && <p>{translate("loading")}</p>}

      {state.status === "anonymous" && (
        // A plain link, not a fetch: the login route answers with a redirect to
        // Discord, which has to be a top-level navigation for the consent
        // screen to appear and for the state cookie to be set on this origin.
        <a className="button" href="/api/auth/login">
          {translate("login_button")}
        </a>
      )}

      {state.status === "error" && (
        <div>
          <p className="error">
            {translate(
              state.code === "discord_unavailable"
                ? "error_discord_unavailable"
                : "error_generic",
            )}
          </p>
          <button type="button" className="button" onClick={() => void load()}>
            {translate("retry_button")}
          </button>
        </div>
      )}

      {state.status === "signed-in" && (
        <section>
          <h2>{translate("guilds_heading")}</h2>
          {state.guilds.length === 0 ? (
            <div className="empty">
              <h3>{translate("guilds_empty_title")}</h3>
              <p>{translate("guilds_empty_body")}</p>
            </div>
          ) : (
            <ul className="guild-list">
              {state.guilds.map((guild) => {
                const iconUrl = guildIconUrl(guild);
                return (
                  <li key={guild.id} className="guild">
                    {iconUrl ? (
                      // eslint-disable-next-line @next/next/no-img-element -- a
                      // 40px avatar needs no optimisation pipeline, and next/image
                      // would add a server-side fetch of a third-party URL.
                      <img
                        className="guild-icon"
                        src={iconUrl}
                        alt={translate("guild_icon_alt", { name: guild.name })}
                        width={40}
                        height={40}
                      />
                    ) : (
                      <span className="guild-initial" aria-hidden="true">
                        {guildInitial(guild.name)}
                      </span>
                    )}
                    <span className="guild-name">{guild.name}</span>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      )}
    </main>
  );
}
