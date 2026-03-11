from __future__ import annotations

import ast
from functools import lru_cache
from hashlib import sha256
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

TRAIN_MLX_PRESET_NODES = (
    "RunPreset",
    "PRESETS",
    "DEFAULT_PRESET",
    "default_canonical_eval_batch_size",
)

EVAL_SIGNATURE_FILES = (
    ("autoresearch_mlx/constants.py", None),
    ("autoresearch_mlx/data.py", None),
    ("autoresearch_mlx/train.py", TRAIN_MLX_PRESET_NODES),
)

RUNTIME_SIGNATURE_FILES = (
    ("autoresearch_mlx/constants.py", None),
    ("autoresearch_mlx/data.py", None),
    ("autoresearch_mlx/model.py", None),
    ("autoresearch_mlx/optim.py", None),
    ("autoresearch_mlx/train.py", TRAIN_MLX_PRESET_NODES),
)


def _node_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _module_ast_dump(relative_path: str) -> str:
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    return ast.dump(tree, include_attributes=False)


def _selected_node_dump(relative_path: str, names: tuple[str, ...]) -> str:
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    selected_nodes: list[ast.AST] = []
    requested = set(names)
    for node in tree.body:
        name = _node_name(node)
        if name in requested:
            selected_nodes.append(node)
    module = ast.Module(body=selected_nodes, type_ignores=[])
    return ast.dump(module, include_attributes=False)


def _signature_for(specs: tuple[tuple[str, tuple[str, ...] | None], ...]) -> str:
    digest = sha256()
    for relative_path, selected_names in specs:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        if selected_names is None:
            payload = _module_ast_dump(relative_path)
        else:
            payload = _selected_node_dump(relative_path, selected_names)
        digest.update(payload.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def current_eval_semantics_signature() -> str:
    return _signature_for(EVAL_SIGNATURE_FILES)


@lru_cache(maxsize=1)
def current_runtime_shape_signature() -> str:
    return _signature_for(RUNTIME_SIGNATURE_FILES)


def current_calibration_signatures() -> dict[str, str]:
    return {
        "eval_semantics_signature": current_eval_semantics_signature(),
        "runtime_shape_signature": current_runtime_shape_signature(),
    }
