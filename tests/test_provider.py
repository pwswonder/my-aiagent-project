from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from paper_agent_v2.providers.openai_provider import OpenAIProvider


class ExampleSchema(BaseModel):
    params: dict[str, object]


def test_openai_structured_output_allows_open_parameter_objects() -> None:
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text='{"params":{"width":64}}')

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = SimpleNamespace(responses=Responses())
    provider._model = "test-model"
    provider._embedding_model = "test-embedding"

    result = provider.generate_structured(ExampleSchema, instructions="extract", prompt="paper")

    assert result.params == {"width": 64}
    assert captured["text"]["format"]["strict"] is False


def test_openai_structured_output_retries_application_validation_errors() -> None:
    prompts = []

    class Responses:
        def create(self, **kwargs):
            prompts.append(kwargs["input"])
            output = '{"params":"invalid"}' if len(prompts) == 1 else '{"params":{"width":64}}'
            return SimpleNamespace(output_text=output)

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._client = SimpleNamespace(responses=Responses())
    provider._model = "test-model"
    provider._embedding_model = "test-embedding"

    result = provider.generate_structured(ExampleSchema, instructions="extract", prompt="paper")

    assert result.params == {"width": 64}
    assert len(prompts) == 2
    assert "VALIDATION_ERROR" in prompts[1]
