FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends docker.io && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN mkdir -p /app/var/storage && chown -R 65532:65532 /app/var
CMD ["uvicorn", "paper_agent_v2.sandbox_runner:app", "--host", "0.0.0.0", "--port", "8090"]
