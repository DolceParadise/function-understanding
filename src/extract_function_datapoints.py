#!/usr/bin/env python3

"""Extract function-level datapoints from rcutils C sources.

The script uses ctags to enumerate function definitions, then uses a small
brace-aware scanner to recover the full function body from the source file.
For each extracted function it emits a JSONL datapoint with lightweight labels
derived from the code body.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

from llm import Rits


CONTROL_FLOW_KEYWORDS = ("if", "for", "while", "return")
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "configs" / "system_prompt.txt"
LABEL_RULES_PATH = Path(__file__).resolve().parents[1] / "configs" / "label_rules.json"
DEFAULT_LLM_MODEL = os.environ.get("RITS_MODEL", "moonshotai/Kimi-K2.5")


@dataclass(frozen=True)
class FunctionRecord:
    function_name: str
    function_code: str
    file_path: str
    labels: dict


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
        default=Path(__file__).resolve().parents[1] / "rcutils",
        help="Directory to scan for C files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "rcutils_function_datapoints.jsonl",
        help="JSONL output path.",
    )
    return parser.parse_args()


def iter_c_files(source_root: Path) -> Iterator[Path]:
    for path in sorted(source_root.rglob("*.c")):
        if path.is_file():
            yield path


def run_ctags(file_path: Path) -> list[tuple[str, int, str]]:
    completed = subprocess.run(
        ["ctags", "-x", str(file_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    entries: list[tuple[str, int, str]] = []
    for raw_line in completed.stdout.splitlines():
        match = re.match(r"^(\S+)\s+(\d+)\s+(\S+)\s+(.*)$", raw_line)
        if not match:
            continue
        name, line_number, _source_path, signature = match.groups()
        entries.append((name, int(line_number), signature))
    return entries


def build_line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer(r"\n", text):
        offsets.append(match.end())
    return offsets


def line_start_offset(line_offsets: list[int], line_number: int) -> int:
    return line_offsets[max(0, line_number - 1)]


def offset_to_line_number(line_offsets: list[int], offset: int) -> int:
    # line_offsets stores the start offset for each line (1-based line index).
    low = 0
    high = len(line_offsets) - 1
    while low <= high:
        mid = (low + high) // 2
        if line_offsets[mid] <= offset:
            low = mid + 1
        else:
            high = mid - 1
    return high + 1


def find_function_body(text: str, start_offset: int) -> tuple[int, int]:
    in_block_comment = False
    in_line_comment = False
    in_string = False
    in_char = False
    escape = False
    brace_depth = 0
    body_open = -1

    index = start_offset
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if in_char:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "'":
                in_char = False
            index += 1
            continue

        if char == "/" and next_char == "/":
            in_line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            in_block_comment = True
            index += 2
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if char == "'":
            in_char = True
            index += 1
            continue

        if char == "{":
            brace_depth += 1
            if body_open < 0:
                body_open = index
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0 and body_open >= 0:
                return body_open, index + 1

        index += 1

    raise ValueError("unbalanced braces while extracting function body")


def find_signature_start_line(lines: list[str], body_open_line: int, function_name_line: int) -> int:
    start_line = function_name_line
    current = function_name_line - 1

    while current >= 1:
        stripped = lines[current - 1].strip()
        if not stripped:
            break
        if stripped.startswith("#"):
            break
        if stripped.endswith(";") or stripped.endswith("}"):
            break
        start_line = current
        current -= 1

    # If the opening brace is on the same line as the signature, ensure the
    # extracted slice still begins at the earliest signature line.
    if body_open_line < start_line:
        return body_open_line
    return start_line


def infer_high_level_purpose(function_name: str, function_code: str) -> str:
    raw_purpose = query_high_level_purpose(function_name, function_code)
    purpose = normalize_high_level_purpose(raw_purpose)
    if purpose:
        return purpose
    
    return fallback_high_level_purpose(function_name, function_code)


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def load_label_rules() -> dict:
    return json.loads(LABEL_RULES_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_llm_client() -> Rits:
    return Rits(model=DEFAULT_LLM_MODEL)


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
    return get_llm_client().generate_text(prompt, max_tokens=64)


def normalize_high_level_purpose(raw_purpose: str) -> str:
    purpose = raw_purpose.strip()
    if not purpose:
        return ""
    if purpose.startswith("Error:"):
        return ""
    if purpose.startswith("{") and purpose.endswith("}"):
        try:
            payload = json.loads(purpose)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            json_value = payload.get("high_level_purpose")
            if isinstance(json_value, str) and json_value.strip():
                purpose = json_value.strip()
    quoted_matches = re.findall(r'"([^"]{3,200})"|\'([^\']{3,200})\'', purpose)
    if quoted_matches:
        for double_quoted, single_quoted in reversed(quoted_matches):
            candidate = double_quoted or single_quoted
            if candidate.strip():
                purpose = candidate.strip()
                break
    lower_purpose = purpose.lower()
    for marker in ("high_level_purpose:", "purpose:", "phrase:", "answer:", "label:"):
        marker_index = lower_purpose.rfind(marker)
        if marker_index != -1:
            purpose = purpose[marker_index + len(marker) :].strip()
            lower_purpose = purpose.lower()
    purpose = purpose.removeprefix("Purpose:").removeprefix("purpose:").strip()
    purpose = purpose.strip('"\'`')
    purpose = re.sub(r"\s+", " ", purpose)
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
    if re.search(r"\b(static|return|void|size_t|const|struct|while|typedef|include)\b", purpose, re.IGNORECASE):
        return False
    word_count = len(purpose.split())
    if word_count < 3 or word_count > 12:
        return False
    return True


def fallback_high_level_purpose(function_name: str, function_code: str) -> str:
    lowered_name = function_name.lower()
    compact_name = lowered_name.removeprefix("rcutils_").removeprefix("__")
    compact_name = compact_name.replace("__", "_")
    code = function_code.lower()
    rules = load_label_rules()

    matched = match_rule(compact_name, rules.get("purpose_rules", []))
    if matched:
        return matched

    side_effects = detect_side_effects(code)
    if side_effects:
        signal_text = ", ".join(side_effects)
        return f"handles {signal_text} concerns for {compact_name.replace('_', ' ')}"

    return build_fallback_purpose(compact_name, rules.get("purpose_fallback_verbs", {}))


def match_rule(compact_name: str, rules: list[dict]) -> str:
    for rule in rules:
        if rule_matches(compact_name, rule):
            return rule["purpose"]
    return ""


def rule_matches(compact_name: str, rule: dict) -> bool:
    tokens = compact_name.split("_") if compact_name else []
    required = rule.get("all", [])
    optional = rule.get("any", [])
    if required and not all(any(fragment in token or fragment in compact_name for token in tokens) for fragment in required):
        return False
    if optional and not any(any(fragment in token or fragment in compact_name for token in tokens) for fragment in optional):
        return False
    return True


def detect_side_effects(code: str) -> list[str]:
    rules = load_label_rules().get("side_effect_patterns", {})
    labels = [label for label, patterns in rules.items() if any(re.search(pattern, code) for pattern in patterns)]
    return labels


def build_fallback_purpose(compact_name: str, verb_map: dict[str, str]) -> str:
    tokens = compact_name.split("_") if compact_name else [compact_name]
    verb = tokens[0] if tokens else compact_name
    remainder = " ".join(tokens[1:]).strip()
    if verb in verb_map:
        return f"{verb_map[verb]} {remainder}".strip()
    return f"operates on {compact_name.replace('_', ' ').strip()}".strip()


def infer_control_flow_elements(function_code: str) -> list[str]:
    elements = [kw for kw in CONTROL_FLOW_KEYWORDS if re.search(rf"\b{kw}\b", function_code)]
    return elements


def infer_side_effects(function_code: str) -> list[str]:
    labels = detect_side_effects(function_code)
    return labels if labels else ["none"]


def extract_function_records(file_path: Path) -> list[FunctionRecord]:
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    line_offsets = build_line_offsets(text)

    records: list[FunctionRecord] = []
    for function_name, definition_line, _signature in run_ctags(file_path):
        start_offset = line_start_offset(line_offsets, definition_line)
        body_open_offset, body_close_offset = find_function_body(text, start_offset)
        body_open_line = offset_to_line_number(line_offsets, body_open_offset)
        body_close_line = offset_to_line_number(line_offsets, body_close_offset)
        start_line = find_signature_start_line(lines, body_open_line, definition_line)
        function_code = "".join(lines[start_line - 1 : body_close_line])

        labels = {
            "high_level_purpose": infer_high_level_purpose(function_name, function_code),
            "control_flow_elements": infer_control_flow_elements(function_code),
            "side_effects": infer_side_effects(function_code),
        }
        records.append(
            FunctionRecord(
                function_name=function_name,
                function_code=function_code,
                file_path=str(file_path),
                labels=labels,
            )
        )
    return records


def validate_record(record: FunctionRecord) -> None:
    if not record.function_name:
        raise ValueError("missing function_name")
    if not record.function_code.strip():
        raise ValueError(f"empty function_code for {record.function_name}")
    if record.labels["side_effects"] == []:
        raise ValueError(f"missing side_effects for {record.function_name}")
    for element in record.labels["control_flow_elements"]:
        if element not in CONTROL_FLOW_KEYWORDS:
            raise ValueError(f"unexpected control flow label {element!r}")


def write_jsonl(records: Iterable[FunctionRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_record(record)
            payload = {
                "function_name": record.function_name,
                "function_code": record.function_code,
                "file_path": record.file_path,
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