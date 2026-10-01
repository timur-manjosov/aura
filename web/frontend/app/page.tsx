"use client";

/**
 * The whole shell: sign in, see who you are, see the servers you can manage --
 * and, since Phase 4c, each server's plan with the one billing action that
 * fits it.
 *
 * One client component on purpose. Everything on this page depends on a
 * session cookie the server component tree cannot read without forwarding
 * headers, so the page loads, asks the backend who it is talking to, and
 * renders one of three states. When 4d adds real content, that is the point to
 * split.
 *
 * Billing never decides anything here. Whether a server may subscribe, who may
 * open the billing portal and what a plan is worth are all answered by the
 * backend (and by the bot behind it); this page only shows those answers and
 * sends the browser to the Stripe URL the backend returns. Plan information
 * failing to load leaves the server list intact -- signing in and seeing your
 * servers must not depend on billing being reachable.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  fetchBillingGuilds,
  fetchCurrentUser,
  fetchManageableGuilds,
  guildIconUrl,
  guildInitial,
  logout,
  openBillingPortal,
  startCheckout,
  type CurrentUser,
  type GuildPlan,
  type ManageableGuild,
} from "@/lib/api";
import { DEFAULT_LOCALE, resolveLocale, t } from "@/lib/locales";

type LoadState =
  | { status: "loading" }
  | { status: "anonymous" }
  | {
      status: "signed-in";
      user: CurrentUser;
      guilds: ManageableGuild[];
      // null: plan information could not be loaded; the list still renders.
      plans: Record<string, GuildPlan> | null;
    }
  | { status: "error"; code: string };

type Translate = (key: string, params?: Record<string, string | number>) => string;

/** Backend error codes a billing action can return, mapped to their catalogue keys. */
const BILLING_ERROR_KEYS: Record<string, string> = {
  discord_unavailable: "error_discord_unavailable",
  guild_not_manageable: "error_guild_not_manageable",
  already_subscribed: "error_already_subscribed",
  nothing_to_buy: "error_nothing_to_buy",
  not_billing_owner: "error_not_billing_owner",
  payment_provider_unavailable: "error_payment_provider",
  payment_provider_error: "error_payment_provider",
  billing_unavailable: "billing_unavailable",
  rate_limited: "error_rate_limited",
};

/** Backend error codes a page load can return with their own message; anything else is generic. */
const LOAD_ERROR_KEYS: Record<string, string> = {
  discord_unavailable: "error_discord_unavailable",
  rate_limited: "error_rate_limited",
};

const CHECKOUT_NOTICES: Record<string, string> = {
  success: "checkout_success_notice",
  cancelled: "checkout_cancelled_notice",
};

function formatDate(seconds: number, locale: string): string {
  const moment = new Date(seconds * 1000);
  try {
    return new Intl.DateTimeFormat(locale, { dateStyle: "medium", timeStyle: "short" }).format(moment);
  } catch {
    return new Intl.DateTimeFormat(DEFAULT_LOCALE, { dateStyle: "medium", timeStyle: "short" }).format(moment);
  }
}

/** The one sentence describing a plan, or null when there is nothing to add to the badge. */
function planDescription(plan: GuildPlan, translate: Translate, locale: string): string | null {
  if (plan.basis === "billing_not_enforced") {
    return translate("plan_basis_not_enforced");
  }
  if (plan.basis === "complimentary") {
    return translate("plan_basis_complimentary");
  }
  const dated = (key: string, seconds: number | null) =>
    seconds === null ? null : translate(key, { date: formatDate(seconds, locale) });
  switch (plan.standing) {
    case "active":
      return dated("plan_standing_active", plan.paid_through);
    case "renewal_pending":
      return dated("plan_standing_renewal_pending", plan.access_until);
    case "payment_pending":
      return translate("plan_standing_payment_pending");
    case "canceling":
      return dated("plan_standing_canceling", plan.access_until);
    case "payment_grace":
      return dated("plan_standing_payment_grace", plan.access_until);
    case "ended":
      return translate("plan_standing_ended");
    default:
      return null;
  }
}

export default function HomePage() {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [locale, setLocale] = useState<string>(DEFAULT_LOCALE);
  const [notice, setNotice] = useState<string | null>(null);
  const [busyGuild, setBusyGuild] = useState<string | null>(null);
  const [actionError, setActionError] = useState<{ guildId: string; key: string } | null>(null);

  // Read the browser's language and the checkout return parameter after
  // mount, never during render: the server has neither, and reading them
  // during render would make the first client paint disagree with the HTML.
  useEffect(() => {
    setLocale(resolveLocale(typeof navigator === "undefined" ? null : navigator.language));
    const outcome = new URLSearchParams(window.location.search).get("checkout");
    if (outcome !== null && outcome in CHECKOUT_NOTICES) {
      setNotice(CHECKOUT_NOTICES[outcome]);
    }
    if (outcome !== null) {
      // Drop the parameter so a reload does not show the notice again.
      window.history.replaceState(null, "", window.location.pathname);
    }
  }, []);

  const translate = useMemo<Translate>(
    () => (key, params) => t(key, locale, params),
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

    let plans: Record<string, GuildPlan> | null = null;
    if (guilds.value.length > 0) {
      const billing = await fetchBillingGuilds();
      if (billing.kind === "ok") {
        plans = Object.fromEntries(billing.value.map((entry) => [entry.id, entry.plan]));
      }
    } else {
      plans = {};
    }
    setState({ status: "signed-in", user: me.value, guilds: guilds.value, plans });
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const signOut = useCallback(async () => {
    await logout();
    setState({ status: "anonymous" });
  }, []);

  const redirectForBilling = useCallback(
    async (guildId: string, action: typeof startCheckout) => {
      setBusyGuild(guildId);
      setActionError(null);
      const result = await action(guildId);
      if (result.kind === "ok") {
        // A top-level navigation: Stripe's hosted pages do not run in a frame.
        window.location.assign(result.value.url);
        return;
      }
      setBusyGuild(null);
      setActionError({ guildId, key: BILLING_ERROR_KEYS[result.code] ?? "error_generic" });
    },
    [],
  );

  const anyFreeGuild =
    state.status === "signed-in" &&
    state.plans !== null &&
    Object.values(state.plans).some((plan) => plan.tier === "free");

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
      {notice !== null && <p className="notice success">{translate(notice)}</p>}

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
            {translate(LOAD_ERROR_KEYS[state.code] ?? "error_generic")}
          </p>
          <button type="button" className="button" onClick={() => void load()}>
            {translate("retry_button")}
          </button>
        </div>
      )}

      {state.status === "signed-in" && (
        <section>
          <h2>{translate("guilds_heading")}</h2>
          {state.plans === null && <p className="error">{translate("billing_unavailable")}</p>}
          {anyFreeGuild && <p className="tagline">{translate("pro_features_note")}</p>}
          {state.guilds.length === 0 ? (
            <div className="empty">
              <h3>{translate("guilds_empty_title")}</h3>
              <p>{translate("guilds_empty_body")}</p>
            </div>
          ) : (
            <ul className="guild-list">
              {state.guilds.map((guild) => {
                const iconUrl = guildIconUrl(guild);
                const plan = state.plans?.[guild.id] ?? null;
                const description = plan ? planDescription(plan, translate, locale) : null;
                const busy = busyGuild === guild.id;
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
                    <div className="guild-body">
                      <span className="guild-name">
                        {guild.name}
                        {plan && (
                          <span className={`plan-badge ${plan.tier}`}>
                            {translate(plan.tier === "pro" ? "plan_badge_pro" : "plan_badge_free")}
                          </span>
                        )}
                      </span>
                      {description && (
                        <span className={`plan-line ${plan?.standing === "payment_grace" ? "warning" : ""}`}>
                          {description}
                        </span>
                      )}
                      {plan && plan.active_subscription_count > 1 && (
                        <span className="plan-line warning">
                          {translate("plan_multiple_subscriptions", { count: plan.active_subscription_count })}
                        </span>
                      )}
                      {actionError?.guildId === guild.id && (
                        <span className="plan-line error">{translate(actionError.key)}</span>
                      )}
                    </div>
                    {plan && (
                      <div className="guild-actions">
                        {plan.can_subscribe && (
                          <button
                            type="button"
                            className="button"
                            disabled={busyGuild !== null}
                            onClick={() => void redirectForBilling(guild.id, startCheckout)}
                          >
                            {translate(busy ? "redirecting" : "subscribe_button")}
                          </button>
                        )}
                        {plan.is_billing_owner && (
                          <button
                            type="button"
                            className="button secondary"
                            disabled={busyGuild !== null}
                            onClick={() => void redirectForBilling(guild.id, openBillingPortal)}
                          >
                            {translate(busy ? "redirecting" : "manage_billing_button")}
                          </button>
                        )}
                      </div>
                    )}
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
