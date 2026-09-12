FROM python:3.12-slim

# Same shape as the bot's own Dockerfile: a fixed non-root uid created before
# anything is copied, so no layer ever runs or is owned by root.
RUN groupadd --gid 1001 aurraweb \
    && useradd --uid 1001 --gid aurraweb --create-home --shell /usr/sbin/nologin aurraweb

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN chown -R aurraweb:aurraweb /app
USER aurraweb

# Copied after the dependency layer so editing source during development does
# not re-run pip install.
COPY --chown=aurraweb:aurraweb aura_web/ ./aura_web/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    AURA_WEB_HOST=0.0.0.0 \
    AURA_WEB_PORT=8080

EXPOSE 8080

# The healthcheck hits /api/health, which deliberately reaches nothing
# external -- a check that called Discord would mark this container unhealthy
# during somebody else's outage and have the orchestrator restart a process
# that is working perfectly.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["python", "-m", "aura_web"]
