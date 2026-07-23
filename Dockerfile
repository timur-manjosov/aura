FROM python:3.12-slim

RUN groupadd --gid 1000 aura \
    && useradd --uid 1000 --gid aura --create-home --shell /usr/sbin/nologin aura

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
RUN chown -R aura:aura /app

USER aura

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "aura.main"]
