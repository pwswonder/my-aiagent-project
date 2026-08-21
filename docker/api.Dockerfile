FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md alembic.ini ./
COPY migrations ./migrations
COPY src ./src
COPY v2_app.py ./v2_app.py
RUN pip install --no-cache-dir . streamlit
RUN mkdir -p /app/var/storage && chown -R 65532:65532 /app/var
USER 65532:65532
CMD ["paper-agent-api"]
