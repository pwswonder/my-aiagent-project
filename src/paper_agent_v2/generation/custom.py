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


class CustomModuleResponse(BaseModel):
    class_name: str = Field(pattern=r"^[A-Z][A-Za-z0-9_]*$")
    source: str
    assumptions: list[str] = Field(default_factory=list)


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
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is forbidden")


def synthesize_custom_module(provider: LLMProvider, node: NodeSpec) -> CustomModuleResponse:
    contract = {
        "node_id": node.id,
        "operation": node.op,
        "inputs": node.inputs,
        "output": node.output,
        "parameters": node.params,
    }
    response = provider.generate_structured(
        CustomModuleResponse,
        instructions=(
            "Generate exactly one self-contained torch.nn.Module class for the supplied contract. "
            "Imports are limited to torch, typing and math. Do not perform file, network, process, "
            "environment, reflection or dynamic-code operations. Preserve tensor shape semantics."
        ),
        prompt=str(contract),
    )
    response.source = re.sub(r"^```(?:python)?\s*|\s*```$", "", response.source.strip())
    validate_custom_module(response.source, response.class_name)
    return response
