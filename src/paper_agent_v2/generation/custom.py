from __future__ import annotations

import ast
import re

from pydantic import BaseModel, Field

from paper_agent_v2.ir import NodeSpec
from paper_agent_v2.providers.base import LLMProvider

FORBIDDEN_NAMES = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "__import__",
}
ALLOWED_IMPORT_ROOTS = {"torch", "typing", "math"}


def _is_safe_super_init(node: ast.Attribute) -> bool:
    return (
        node.attr == "__init__"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "super"
        and not node.value.args
        and not node.value.keywords
    )


class CustomModuleResponse(BaseModel):
    class_name: str = Field(pattern=r"^[A-Z][A-Za-z0-9_]*$")
    source: str
    assumptions: list[str] = Field(default_factory=list)


class CustomModuleBatchItem(CustomModuleResponse):
    node_id: str


class CustomModuleBatchResponse(BaseModel):
    modules: list[CustomModuleBatchItem]


def validate_custom_module(source: str, class_name: str) -> None:
    if len(source) > 20_000:
        raise ValueError("custom module source is too large")
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1 or classes[0].name != class_name:
        raise ValueError("custom source must contain exactly the requested class")
    bases = {ast.unparse(base) for base in classes[0].bases}
    if not bases.intersection({"nn.Module", "torch.nn.Module"}):
        raise ValueError("custom class must subclass nn.Module")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            module = node.module if isinstance(node, ast.ImportFrom) else names[0]
            root = (module or names[0]).split(".")[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"forbidden import: {root}")
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name.split(".")[-1] in FORBIDDEN_NAMES:
                raise ValueError(f"forbidden call: {name}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and not _is_safe_super_init(node):
            raise ValueError("dunder attribute access is forbidden")


def synthesize_custom_module(provider: LLMProvider, node: NodeSpec) -> CustomModuleResponse:
    contract = {
        "node_id": node.id,
        "operation": node.op,
        "inputs": node.inputs,
        "output": node.output,
        "parameters": node.params,
    }
    instructions = (
        "Generate exactly one self-contained torch.nn.Module class for the supplied contract. "
        "The class_name field must exactly match the single class declared in source. "
        "Imports are limited to torch, typing and math. Do not perform file, network, process, "
        "environment, reflection or dynamic-code operations. Preserve tensor shape semantics."
    )
    validation_error = ""
    for _ in range(3):
        prompt = (
            f"{contract}\nPrevious response validation error: {validation_error}"
            if validation_error
            else str(contract)
        )
        response = provider.generate_structured(
            CustomModuleResponse,
            instructions=instructions,
            prompt=prompt,
        )
        response.source = re.sub(r"^```(?:python)?\s*|\s*```$", "", response.source.strip())
        try:
            validate_custom_module(response.source, response.class_name)
        except (SyntaxError, ValueError) as exc:
            validation_error = str(exc)
            continue
        return response
    raise ValueError(f"custom module generation failed validation after 3 attempts: {validation_error}")


def synthesize_custom_modules(
    provider: LLMProvider, nodes: list[NodeSpec]
) -> dict[str, CustomModuleResponse]:
    """Generate all novel blocks in one bounded request instead of N serial calls."""
    pending = {node.id: node for node in nodes}
    generated: dict[str, CustomModuleResponse] = {}
    validation_errors: dict[str, str] = {}
    instructions = (
        "Generate one independent, self-contained torch.nn.Module for every supplied node contract. "
        "Return each requested node_id exactly once. The class_name must match the single class in source. "
        "Imports are limited to torch, typing and math. Do not perform file, network, process, environment, "
        "reflection or dynamic-code operations. Preserve tensor shape semantics."
    )
    for _ in range(3):
        contracts = [
            {
                "node_id": node.id,
                "operation": node.op,
                "inputs": node.inputs,
                "output": node.output,
                "parameters": node.params,
                "previous_validation_error": validation_errors.get(node.id),
            }
            for node in pending.values()
        ]
        response = provider.generate_structured(
            CustomModuleBatchResponse,
            instructions=instructions,
            prompt=str(contracts),
        )
        returned_ids: set[str] = set()
        for item in response.modules:
            if item.node_id not in pending or item.node_id in returned_ids:
                continue
            returned_ids.add(item.node_id)
            item.source = re.sub(r"^```(?:python)?\s*|\s*```$", "", item.source.strip())
            try:
                validate_custom_module(item.source, item.class_name)
            except (SyntaxError, ValueError) as exc:
                validation_errors[item.node_id] = str(exc)
                continue
            generated[item.node_id] = CustomModuleResponse(
                class_name=item.class_name,
                source=item.source,
                assumptions=item.assumptions,
            )
        pending = {node_id: node for node_id, node in pending.items() if node_id not in generated}
        for node_id in pending:
            validation_errors.setdefault(node_id, "module was missing from the batch response")
        if not pending:
            return generated
    details = "; ".join(f"{node_id}: {error}" for node_id, error in validation_errors.items())
    raise ValueError(f"custom module batch generation failed after 3 attempts: {details}")
