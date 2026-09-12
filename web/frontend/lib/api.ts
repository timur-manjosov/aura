/**
 * The browser's view of the backend.
 *
 * Every call is same-origin (`/api/...`, proxied by next.config.mjs) and sends
 * the session cookie via `credentials: "same-origin"`. There is no token
 * handling here, and there is nothing to add later: the browser never receives
 * a Discord token, so there is nothing for this module to store, refresh or
 * attach to a header.
 *
 * Errors are returned as backend error CODES, never as prose. The page turns a
 * code into text through the translation catalogue, which is what keeps
 * user-facing strings out of logic -- CLAUDE.md's rule, applied across the
 * process boundary.
 */

export type ApiError = { kind: "error"; code: string };
export type ApiOk<T> = { kind: "ok"; value: T };
export type ApiResult<T> = ApiOk<T> | ApiError;

export type CurrentUser = {
  id: string;
  username: string;
  global_name: string | null;
  avatar: string | null;
};

export type ManageableGuild = {
  id: string;
  name: string;
  icon: string | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<ApiResult<T>> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      credentials: "same-origin",
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    // A rejected fetch is the network, not the backend: there is no code to
    // read off a response that never arrived.
    return { kind: "error", code: "network_unreachable" };
  }

  if (response.status === 204) {
    return { kind: "ok", value: undefined as T };
  }

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    const code =
      typeof body === "object" && body !== null && typeof (body as { error?: unknown }).error === "string"
        ? (body as { error: string }).error
        : "error_generic";
    return { kind: "error", code };
  }
  return { kind: "ok", value: body as T };
}

export function fetchCurrentUser(): Promise<ApiResult<CurrentUser>> {
  return request<CurrentUser>("/api/me");
}

export function fetchManageableGuilds(): Promise<ApiResult<ManageableGuild[]>> {
  return request<ManageableGuild[]>("/api/guilds");
}

export function logout(): Promise<ApiResult<void>> {
  // POST, not GET: a logout reachable by navigation is a logout any page can
  // trigger with an <img> tag.
  return request<void>("/api/auth/logout", { method: "POST" });
}

/** Build the CDN URL for a guild icon whose hash the backend already validated. */
export function guildIconUrl(guild: ManageableGuild): string | null {
  return guild.icon
    ? `https://cdn.discordapp.com/icons/${guild.id}/${guild.icon}.png?size=64`
    : null;
}

/** The first letter of a server's name, for the placeholder when it has no icon. */
export function guildInitial(name: string): string {
  // Array.from, not name[0]: an emoji or any astral-plane character is two
  // UTF-16 code units, and slicing one off renders as a replacement glyph.
  return Array.from(name)[0]?.toUpperCase() ?? "?";
}
