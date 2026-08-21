from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMProvider(Protocol):
    @property
    def identity(self) -> str: ...

    def generate_structured(
        self,
        schema: type[SchemaT],
        *,
        instructions: str,
        prompt: str,
    ) -> SchemaT: ...

    def generate_text(self, *, instructions: str, prompt: str) -> str: ...

    def analyze_images(
        self,
        image_paths: list[Path],
        *,
        instructions: str,
        prompt: str,
    ) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...
