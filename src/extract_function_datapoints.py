#!/usr/bin/env python3

"""Extract function-level datapoints from rcutils C sources.

The script uses tree-sitter to parse C sources, recover function definitions,
and derive AST-based features for each function. For each extracted function it
emits a JSONL datapoint with lightweight labels derived from the code body and
an LLM-backed high-level purpose label.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator
from tqdm import tqdm
from openRouter import OpenRouter, NvidiaNIM


SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "configs" / "system_prompt.txt"
LABEL_RULES_PATH = Path(__file__).resolve().parents[1] / "configs" / "label_rules.json"
DEFAULT_LLM_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-31b-it")


@dataclass(frozen=True)
class FunctionRecord:
    function_name: str
    function_code: str
    file_path: str
    ast_features: dict
    labels: dict


@dataclass(frozen=True)
class AstFeatureConfig:
    control_flow_elements: tuple[str, ...]
    statement_distribution_labels: tuple[str, ...]
    statement_node_labels: dict[str, str]
    control_flow_node_labels: dict[str, str]
    cyclomatic_complexity_node_types: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract rcutils C function datapoints into JSONL."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing rcutils/ and data/.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Directory to scan for C files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path.",
    )

    args = parser.parse_args()
    if args.source_root is None:
        args.source_root = args.repo_root / "rcutils"
    if args.output is None:
        args.output = args.repo_root / "data" / "rcutils_function_datapoints.jsonl"
    return args


def iter_c_files(source_root: Path) -> Iterator[Path]:
    for path in sorted(source_root.rglob("*.c")):
        if path.is_file():
            yield path


@lru_cache(maxsize=1)
def get_c_parser():
    from tree_sitter import Parser

    parser = Parser()
    parser.language = load_c_language()
    return parser


@lru_cache(maxsize=1)
def load_c_language():
    from tree_sitter import Language
    import tree_sitter_c

    return Language(tree_sitter_c.language())


def parse_c_source(file_path: Path):
    source_bytes = file_path.read_bytes()
    tree = get_c_parser().parse(source_bytes)
    return source_bytes, tree


def iter_named_descendants(node):
    for child in node.named_children:
        yield child
        yield from iter_named_descendants(child)


def iter_function_definition_nodes(node):
    for child in node.named_children:
        if child.type == "function_definition":
            yield child
        yield from iter_function_definition_nodes(child)


def first_descendant_of_type(node, node_type: str):
    if node.type == node_type:
        return node
    for child in node.named_children:
        match = first_descendant_of_type(child, node_type)
        if match is not None:
            return match
    return None


def get_function_body_node(function_node):
    body_node = function_node.child_by_field_name("body")
    if body_node is not None:
        return body_node
    for child in reversed(function_node.named_children):
        if child.type == "compound_statement":
            return child
    raise ValueError("function definition is missing a body")


def get_function_name(function_node, source_bytes: bytes) -> str:
    declarator = function_node.child_by_field_name("declarator")
    if declarator is None:
        declarator = first_descendant_of_type(function_node, "function_declarator")
    if declarator is None:
        raise ValueError("function definition is missing a declarator")

    identifier = first_descendant_of_type(declarator, "identifier")
    if identifier is None:
        raise ValueError("function definition is missing an identifier")
    return source_bytes[identifier.start_byte : identifier.end_byte].decode("utf-8")


def compute_ast_depth(node) -> int:
    child_depths = [compute_ast_depth(child) for child in node.named_children]
    if not child_depths:
        return 1
    return 1 + max(child_depths)


@lru_cache(maxsize=1)
def load_label_rules() -> dict:
    return json.loads(LABEL_RULES_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_ast_feature_config() -> AstFeatureConfig:
    raw_config = load_label_rules().get("ast_features", {})
    return AstFeatureConfig(
        control_flow_elements=tuple(raw_config.get("control_flow_elements", ())),
        statement_distribution_labels=tuple(raw_config.get("statement_distribution_labels", ())),
        statement_node_labels=dict(raw_config.get("statement_node_labels", {})),
        control_flow_node_labels=dict(raw_config.get("control_flow_node_labels", {})),
        cyclomatic_complexity_node_types=tuple(raw_config.get("cyclomatic_complexity_node_types", ())),
    )


def count_labeled_descendants(
    root_node,
    label_map: dict[str, str],
    ordered_labels: tuple[str, ...],
    include_zero_values: bool = False,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for descendant in iter_named_descendants(root_node):
        label = label_map.get(descendant.type)
        if label is not None:
            counts[label] += 1

    if include_zero_values:
        return {label: counts.get(label, 0) for label in ordered_labels}
    return {label: counts[label] for label in ordered_labels if counts[label] > 0}


def count_statement_distribution(body_node, ast_config: AstFeatureConfig | None = None) -> dict[str, int]:
    ast_config = ast_config or load_ast_feature_config()
    return count_labeled_descendants(
        body_node,
        ast_config.statement_node_labels,
        ast_config.statement_distribution_labels,
    )


def count_control_flow_elements(body_node, ast_config: AstFeatureConfig | None = None) -> dict[str, int]:
    ast_config = ast_config or load_ast_feature_config()
    return count_labeled_descendants(
        body_node,
        ast_config.control_flow_node_labels,
        ast_config.control_flow_elements,
        include_zero_values=True,
    )


def count_pointer_dereferences(body_node) -> int:
    total = 0
    for descendant in iter_named_descendants(body_node):
        if descendant.type != "unary_expression":
            continue
        if any(child.type == "*" for child in descendant.children):
            total += 1
    return total


def count_cyclomatic_complexity(body_node, ast_config: AstFeatureConfig | None = None) -> int:
    ast_config = ast_config or load_ast_feature_config()
    complexity = 1
    for descendant in iter_named_descendants(body_node):
        if descendant.type in ast_config.cyclomatic_complexity_node_types:
            complexity += 1
            continue
        if descendant.type == "binary_expression":
            complexity += sum(1 for child in descendant.children if child.type in {"&&", "||"})
    return complexity


def build_ast_features(function_node, ast_config: AstFeatureConfig | None = None) -> dict:
    ast_config = ast_config or load_ast_feature_config()
    body_node = get_function_body_node(function_node)
    return {
        "ast_depth": compute_ast_depth(function_node),
        "cyclomatic_complexity": count_cyclomatic_complexity(body_node, ast_config),
        "statement_distribution": count_statement_distribution(body_node, ast_config),
        "pointer_dereferences": count_pointer_dereferences(body_node),
    }


def infer_high_level_purpose(function_name: str, function_code: str) -> str:
    raw_purpose = query_high_level_purpose(function_name, function_code)
    print (raw_purpose)
    purpose = normalize_high_level_purpose(raw_purpose)
    print (purpose)
    return purpose


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def get_llm_client() -> NvidiaNIM:
    return NvidiaNIM(model=DEFAULT_LLM_MODEL)


def build_user_prompt(function_name: str, function_code: str) -> str:
    return (
        "Label the purpose of the following C function.\n\n"
        f"Function name: {function_name}\n"
        "Function code:\n"
        "```c\n"
        f"{function_code.rstrip()}\n"
        "```\n\n"
        "Return only valid JSON with one string field named high_level_purpose."
    )


def build_llm_prompt(function_name: str, function_code: str) -> str:
    system_prompt = load_system_prompt()
    user_prompt = build_user_prompt(function_name, function_code)
    return f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\nASSISTANT:\n"


def query_high_level_purpose(function_name: str, function_code: str) -> str:
    prompt = build_llm_prompt(function_name, function_code)
    return (get_llm_client().generate_text(prompt, max_tokens=64))


def normalize_high_level_purpose(raw_purpose: str) -> str:
    purpose = raw_purpose.strip()
    if not purpose:
        return ""
    if purpose.startswith("Error:"):
        return ""
    if purpose.startswith("```"):
        lines = purpose.splitlines()
        if len(lines) >= 3:
            purpose = "\n".join(lines[1:-1]).strip()
    if purpose.startswith("{") and purpose.endswith("}"):
        try:
            payload = json.loads(purpose)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            json_value = payload.get("high_level_purpose")
            if isinstance(json_value, str) and json_value.strip():
                purpose = json_value.strip()
    purpose = purpose.strip().strip('"\'`')
    purpose = purpose.replace("high_level_purpose:", "")
    purpose = purpose.replace("High_level_purpose:", "")
    for marker in ("high_level_purpose:", "purpose:", "phrase:", "answer:", "label:"):
        marker_index = purpose.lower().rfind(marker)
        if marker_index != -1:
            purpose = purpose[marker_index + len(marker) :].strip()
    lines = [line.strip() for line in purpose.splitlines() if line.strip()]
    if lines:
        purpose = lines[-1]
    if len(purpose.split()) > 12 and "." in purpose:
        tail = purpose.rsplit(".", 1)[-1].strip()
        if 3 <= len(tail.split()) <= 12:
            purpose = tail
    if len(purpose) > 120:
        purpose = purpose[:120].rsplit(" ", 1)[0].strip()
    if not is_concise_purpose(purpose):
        return ""
    return purpose


def is_concise_purpose(purpose: str) -> bool:
    if not purpose:
        return False
    if any(marker in purpose for marker in ("{", "}", "(", ")", ";", "[", "]", "```")):
        return False
    if any(ch in purpose for ch in (".", ":", "?", "!", "=", "/", "\\")):
        return False
    words = {word.strip('"\'`.,').lower() for word in purpose.split()}
    if words.intersection({"static", "return", "void", "size_t", "const", "struct", "while", "typedef", "include"}):
        return False
    word_count = len(purpose.split())
    if word_count < 2 or word_count > 12:
        return False
    return True


def detect_side_effects(code: str) -> list[str]:
    code_lower = code.lower()
    rules = load_side_effect_patterns()
    labels = []
    for label, patterns in rules.items():
        if any(identifier_occurs(code_lower, pattern) for pattern in patterns):
            labels.append(label)
    return labels


@lru_cache(maxsize=1)
def load_side_effect_patterns() -> dict[str, list[str]]:
    raw_patterns = load_label_rules().get("side_effect_patterns", {})
    normalized: dict[str, list[str]] = {}
    for label, patterns in raw_patterns.items():
        normalized[label] = [pattern.strip().lower() for pattern in patterns if pattern.strip()]
    return normalized


def identifier_occurs(text: str, needle: str) -> bool:
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return False
        before = text[index - 1] if index > 0 else ""
        after_index = index + len(needle)
        after = text[after_index] if after_index < len(text) else ""
        if not is_identifier_char(before) and not is_identifier_char(after):
            return True
        start = index + 1


def is_identifier_char(character: str) -> bool:
    return character.isalnum() or character == "_"


def extract_function_records(file_path: Path) -> list[FunctionRecord]:
    source_bytes, tree = parse_c_source(file_path)
    source_text = source_bytes.decode("utf-8")
    ast_config = load_ast_feature_config()

    records: list[FunctionRecord] = []
    for function_node in iter_function_definition_nodes(tree.root_node):
        function_name = get_function_name(function_node, source_bytes)
        function_code = source_text[function_node.start_byte : function_node.end_byte]
        ast_features = build_ast_features(function_node, ast_config)
        body_node = get_function_body_node(function_node)

        labels = {
            "high_level_purpose": infer_high_level_purpose(function_name, function_code),
            "control_flow_elements": count_control_flow_elements(body_node, ast_config),
            "side_effects": detect_side_effects(function_code),
        }
        records.append(
            FunctionRecord(
                function_name=function_name,
                function_code=function_code,
                file_path=str(file_path),
                ast_features=ast_features,
                labels=labels,
            )
        )
    return records


def validate_record(record: FunctionRecord) -> None:
    ast_config = load_ast_feature_config()
    if not record.function_name:
        raise ValueError("missing function_name")
    if not record.function_code.strip():
        raise ValueError(f"empty function_code for {record.function_name}")
    ast_features = record.ast_features
    required_ast_keys = {"ast_depth", "cyclomatic_complexity", "statement_distribution", "pointer_dereferences"}
    if set(ast_features) != required_ast_keys:
        raise ValueError(f"unexpected ast_features keys for {record.function_name}")
    if not isinstance(ast_features["statement_distribution"], dict):
        raise ValueError(f"invalid statement_distribution for {record.function_name}")
    if not isinstance(record.labels.get("control_flow_elements"), dict):
        raise ValueError(f"invalid control_flow_elements for {record.function_name}")
    control_flow_elements = record.labels["control_flow_elements"]
    if set(control_flow_elements) != set(ast_config.control_flow_elements):
        raise ValueError(f"unexpected control flow keys for {record.function_name}")
    if not isinstance(record.labels.get("side_effects"), list):
        raise ValueError(f"invalid side_effects for {record.function_name}")


def write_jsonl(records: Iterable[FunctionRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_record(record)
            payload = {
                "function_name": record.function_name,
                "function_code": record.function_code,
                "file_path": record.file_path,
                "ast_features": record.ast_features,
                "labels": record.labels,
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    source_root = args.source_root
    output_path = args.output

    all_records: list[FunctionRecord] = []
    for file_path in iter_c_files(source_root):
        all_records.extend(extract_function_records(file_path))

    write_jsonl(all_records, output_path)
    print(f"wrote {len(all_records)} datapoints to {output_path}")


if __name__ == "__main__":
    main()