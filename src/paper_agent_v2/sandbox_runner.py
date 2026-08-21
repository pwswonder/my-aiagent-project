from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from paper_agent_v2.config import get_settings
from paper_agent_v2.sandbox import DockerSandbox


class ValidationRequest(BaseModel):
    artifact_relative_path: str = Field(min_length=1, max_length=500)


app = FastAPI(title="AI Paper Agent sandbox runner", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/validate")
def validate(request: ValidationRequest) -> dict[str, object]:
    settings = get_settings()
    root = settings.storage_root.resolve()
    package_path = (root / request.artifact_relative_path).resolve()
    if root not in package_path.parents:
        raise HTTPException(status_code=400, detail="artifact path escaped storage root")
    sandbox = DockerSandbox(
        settings.sandbox_image,
        timeout_seconds=settings.sandbox_timeout_seconds,
        memory=settings.sandbox_memory,
        cpus=settings.sandbox_cpus,
        pids=settings.sandbox_pids_limit,
    )
    return sandbox.validate_volume(request.artifact_relative_path, settings.sandbox_storage_volume).as_json()
