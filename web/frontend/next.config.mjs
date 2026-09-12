/**
 * Next.js configuration.
 *
 * The rewrite is the security-relevant part, not a convenience. Proxying
 * /api/* to the backend container means the browser only ever sees ONE
 * origin, so:
 *
 *   - the session cookie can stay SameSite=Lax instead of SameSite=None,
 *     which is the flag that actually stops a third-party page from riding
 *     the session;
 *   - there is no CORS configuration to get wrong, because there is no
 *     cross-origin request to allow;
 *   - the OAuth redirect_uri registered at Discord points at this origin,
 *     so the callback's Set-Cookie lands on the same origin the app runs on.
 *
 * AURA_WEB_BACKEND_ORIGIN is read at BUILD time for a production build, not
 * at server start. Next serialises rewrites into routes-manifest.json during
 * `next build`, so `next start` ignores a value set only in the runtime
 * environment -- verified by running exactly that and watching the proxy dial
 * the default port anyway. Consequences, both handled:
 *
 *   - the Dockerfile takes it as a build ARG and exports it before
 *     `npm run build`, and web/docker-compose.yml passes the internal service
 *     address (http://backend:8080) there. Baking a compose-internal hostname
 *     into the image is right rather than merely acceptable: it is fixed by
 *     the compose file, not by the deployment.
 *   - `next dev` evaluates the config at server start, so local development
 *     against a backend on another port works by setting the variable
 *     normally.
 *
 * `output: "standalone"` is what lets the Docker image ship a minimal server
 * bundle instead of the whole node_modules tree.
 */
const backendOrigin = process.env.AURA_WEB_BACKEND_ORIGIN ?? "http://127.0.0.1:8080";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backendOrigin}/api/:path*` }];
  },
  /**
   * Security headers for the HTML shell.
   *
   * The backend sets its own on every /api response; these cover the document
   * Next serves, which those never touch. The CSP is the one with teeth: this
   * page renders server names that came from Discord, and although React
   * escapes them and the backend strips control characters, a CSP is the
   * layer that still holds if either of those is wrong.
   *
   * `'unsafe-inline'` on script-src is Next's requirement, not a preference --
   * it inlines its hydration bootstrap, and a nonce-based policy needs
   * per-request rendering this static shell does not do. The policy is still
   * worth having: it pins every script, connect and image ORIGIN, which is
   * what stops an injected tag from loading or exfiltrating to somewhere else.
   */
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "no-referrer" },
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline'",
              "style-src 'self' 'unsafe-inline'",
              // Discord's CDN is the only external host: guild and user icons,
              // whose hash the backend has already validated against a strict
              // alphabet (see aura_web.permissions.sanitize_icon_hash).
              "img-src 'self' data: https://cdn.discordapp.com",
              "connect-src 'self'",
              "form-action 'self'",
              "frame-ancestors 'none'",
              "base-uri 'self'",
              "object-src 'none'",
            ].join("; "),
          },
        ],
      },
    ];
  },
};

export default nextConfig;
