# Slim, not alpine: pandas/numpy publish manylinux wheels that alpine's musl
# libc cannot use, so alpine would compile them from source - a much larger
# image and a far slower build, for no gain.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencies first, in their own layer: application code changes on every
# commit, requirements almost never. This keeps the expensive pip install
# cached across rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/

# Run as a non-root user. A container that does not need root should not have
# it - it is the cheapest hardening available.
RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app:app /app
USER app

EXPOSE 8000

# The image ships the same /health the orchestrator would use, so "is it up"
# has one answer rather than two.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

CMD ["uvicorn", "tradetrack.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
