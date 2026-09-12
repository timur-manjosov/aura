# Three stages so the runtime image carries neither the npm cache nor the
# build toolchain: only Next's standalone server bundle and its static assets.
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
# `npm ci` when a lockfile is present (reproducible), `npm install` otherwise
# so a fresh checkout without one still builds.
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Next serialises rewrites into the build output, so the backend's address has
# to be known here rather than at container start -- see the comment in
# next.config.mjs. The default is the compose service name, which is what any
# deployment using web/docker-compose.yml wants.
ARG AURA_WEB_BACKEND_ORIGIN=http://backend:8080
ENV AURA_WEB_BACKEND_ORIGIN=$AURA_WEB_BACKEND_ORIGIN \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN addgroup --system --gid 1001 nextjs \
    && adduser --system --uid 1001 --ingroup nextjs nextjs

# The standalone bundle already contains the minimal node_modules Next needs;
# static assets are copied separately because the standalone output
# deliberately excludes them.
COPY --from=builder --chown=nextjs:nextjs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nextjs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD node -e "fetch('http://127.0.0.1:3000/').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"

CMD ["node", "server.js"]
