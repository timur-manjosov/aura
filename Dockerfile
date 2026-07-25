FROM python:3.12-slim

RUN groupadd --gid 1000 aura \
    && useradd --uid 1000 --gid aura --create-home --shell /usr/sbin/nologin aura

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Chowned here, before switching users, so both the model bake below and
# the later source copy can run as the actual runtime user without
# bouncing back to root.
RUN chown -R aura:aura /app

USER aura

# Bake the default embedding model's weights into the image right after the
# dependency layer and before COPY src/, so editing source code during
# development never invalidates this layer's cache and re-triggers a
# ~0.2GB download -- only a requirements.txt (or base image) change does.
# Runs as the same non-root user the container actually runs as, so it
# resolves to the same on-disk cache directory main.py's setup_hook will
# look in at startup and finds these weights already there. If a
# deployment overrides EMBEDDING_MODEL to something else, that model just
# downloads lazily on its own first use instead of being pre-cached here --
# a performance nicety, not a hard dependency.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY --chown=aura:aura src/ ./src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "aura.main"]
