FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
# Compose workers submit generated packages to sandbox-runner, so PyTorch belongs
# only in the isolated sandbox image and is intentionally omitted here.
RUN python -c "import subprocess,sys,tomllib; data=tomllib.load(open('pyproject.toml','rb')); subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir',*data['project']['dependencies']])"
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
RUN mkdir -p /app/var/storage && chown -R 65532:65532 /app/var
USER 65532:65532
CMD ["paper-agent-worker"]
