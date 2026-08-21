from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from paper_agent_v2.config import Settings, get_settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ProviderConfigurationError(RuntimeError):
    pass


class OpenAIProvider:
    """Responses API adapter for both OpenAI and Azure OpenAI."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client, self._model, self._embedding_model = self._build_client()

    @property
    def identity(self) -> str:
        return f"{self.settings.llm_provider}:{self._model}"

    def _build_client(self) -> tuple[Any, str, str]:
        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderConfigurationError("Install the 'openai' package to use an LLM provider") from exc

        if self.settings.llm_provider == "azure":
            required = {
                "AZURE_OPENAI_API_KEY": self.settings.azure_openai_api_key,
                "AZURE_OPENAI_ENDPOINT": self.settings.azure_openai_endpoint,
                "AZURE_OPENAI_DEPLOYMENT": self.settings.azure_openai_deployment,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ProviderConfigurationError(f"Missing Azure settings: {', '.join(missing)}")
            client = AzureOpenAI(
                api_key=self.settings.azure_openai_api_key,
                azure_endpoint=str(self.settings.azure_openai_endpoint),
                api_version=self.settings.azure_openai_api_version,
            )
            embedding_model = self.settings.azure_openai_embedding_deployment or self.settings.embedding_model
            return client, str(self.settings.azure_openai_deployment), embedding_model

        if not self.settings.openai_api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return (
            OpenAI(api_key=self.settings.openai_api_key),
            self.settings.openai_model,
            self.settings.embedding_model,
        )

    def generate_structured(
        self,
        schema: type[SchemaT],
        *,
        instructions: str,
        prompt: str,
    ) -> SchemaT:
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                }
            },
        )
        return schema.model_validate_json(response.output_text)

    def generate_text(self, *, instructions: str, prompt: str) -> str:
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=prompt,
        )
        return str(response.output_text)

    def analyze_images(
        self,
        image_paths: list[Path],
        *,
        instructions: str,
        prompt: str,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for path in image_paths:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{encoded}",
                    "detail": "high",
                }
            )
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=[{"role": "user", "content": content}],
        )
        return str(response.output_text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._embedding_model, input=texts)
        return [item.embedding for item in response.data]


def build_provider(settings: Settings | None = None) -> OpenAIProvider:
    return OpenAIProvider(settings)


class StaticProvider:
    """Deterministic provider used by tests and offline demos."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses

    @property
    def identity(self) -> str:
        return "static:test"

    def generate_structured(
        self,
        schema: type[SchemaT],
        *,
        instructions: str,
        prompt: str,
    ) -> SchemaT:
        payload = self.responses.get(schema.__name__)
        if payload is None:
            raise KeyError(f"No static response for {schema.__name__}")
        return schema.model_validate(payload)

    def generate_text(self, *, instructions: str, prompt: str) -> str:
        return str(self.responses.get("text", ""))

    def analyze_images(
        self,
        image_paths: list[Path],
        *,
        instructions: str,
        prompt: str,
    ) -> str:
        return str(self.responses.get("images", ""))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]
