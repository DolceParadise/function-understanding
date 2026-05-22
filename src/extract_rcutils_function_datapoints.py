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
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


CONTROL_FLOW_KEYWORDS = ("if", "for", "while", "return")

IO_PATTERNS = (
    r"\bprintf\b",
    r"\bfprintf\b",
    r"\bsprintf\b",
    r"\bsnprintf\b",
    r"\bfputs?\b",
    r"\bfwrite\b",
    r"\bfread\b",
    r"\bfopen\b",
    r"\bfclose\b",
    r"\bread\b",
    r"\bwrite\b",
    r"\bopen\b",
    r"\bclose\b",
    r"\bgetenv\b",
    r"\bsetenv\b",
    r"\bunsetenv\b",
    r"\bdlopen\b",
    r"\bdlsym\b",
    r"\bdlclose\b",
    r"\bstat\b",
    r"\baccess\b",
    r"\bmkdir\b",
    r"\bremove\b",
    r"\brename\b",
    r"\bopendir\b",
    r"\breaddir\b",
    r"\bclosedir\b",
    r"RCUTILS_SAFE_FWRITE_TO_STDERR",
)

MEMORY_PATTERNS = (
    r"\bmalloc\b",
    r"\bcalloc\b",
    r"\brealloc\b",
    r"\bfree\b",
    r"\bmemcpy\b",
    r"\bmemmove\b",
    r"\bmemset\b",
    r"\bstrdup\b",
    r"\bstrncpy\b",
    r"\bstrcpy\b",
    r"\bstrcat\b",
    r"\breallocf\b",
)

HARDWARE_PATTERNS = (
    r"\bclock_gettime\b",
    r"\btime\b",
    r"\blocaltime_r\b",
    r"\blocaltime_s\b",
    r"\blocaltime\b",
    r"\bGetSystemTimeAsFileTime\b",
    r"\bQueryPerformanceCounter\b",
    r"\btimespec_get\b",
)


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
    lowered_name = function_name.lower()
    compact_name = lowered_name.removeprefix("rcutils_").removeprefix("__")
    compact_name = compact_name.replace("__", "_")
    code = function_code.lower()

    if "strdup" in lowered_name:
        return "duplicates a string using the provided allocator"
    if "reallocf" in lowered_name:
        return "reallocates memory and frees the original buffer on failure"
    if "reallocate" in lowered_name or "realloc" in lowered_name:
        return "reallocates memory"
    if "deallocate" in lowered_name:
        return "deallocates memory"
    if "allocate" in lowered_name and "default" in lowered_name:
        return "provides a default memory allocation callback"
    if "zero_initialized" in lowered_name:
        return "returns a zero-initialized data structure"
    if "allocator_is_valid" in lowered_name:
        return "validates that an allocator has all required callbacks"
    if "time_point" in lowered_name and "string" in lowered_name:
        return "formats a time point into a string representation"
    if "logging" in lowered_name and "initialize" in lowered_name:
        return "initializes the logging subsystem"
    if "logging" in lowered_name and "shutdown" in lowered_name:
        return "shuts down the logging subsystem"
    if "env" in lowered_name and ("get" in lowered_name or "set" in lowered_name):
        return "reads or writes an environment variable"
    if "shared_library" in lowered_name:
        return "loads, unloads, or queries a shared library"
    if "hash_map" in lowered_name:
        return "manages a hash map data structure"
    if "string_array" in lowered_name:
        return "manages a string array data structure"
    if "string_map" in lowered_name:
        return "manages a string-to-string map data structure"
    if "process" in lowered_name:
        return "interacts with process state or process identifiers"
    if "find" in lowered_name:
        return "searches for a substring within a buffer"
    if "split" in lowered_name:
        return "splits a string into multiple substrings"
    if "join" in lowered_name:
        return "joins strings into a single buffer"
    if "format_string" in lowered_name:
        return "formats text with allocator-backed buffers"
    if "snprintf" in lowered_name:
        return "formats text into a fixed-size buffer"
    if "qsort" in lowered_name:
        return "sorts data using a comparator"
    if "strcmp" in lowered_name or "strcasecmp" in lowered_name:
        return "compares two strings"
    if "strnlen" in lowered_name:
        return "measures the length of a bounded string"

    body_signals: list[str] = []
    if re.search(r"\b(malloc|calloc|realloc|free|strdup|memcpy|memmove|memset)\b", code):
        body_signals.append("memory")
    if re.search(r"\b(fopen|fclose|fprintf|printf|snprintf|fread|fwrite|getenv|setenv|unsetenv|dlopen|dlsym|dlclose)\b", code):
        body_signals.append("io")
    if re.search(r"\b(clock_gettime|time\(|localtime_r|localtime_s|localtime|timespec_get)\b", code):
        body_signals.append("hardware")

    if body_signals:
        signal_text = ", ".join(body_signals)
        return f"handles {signal_text} concerns for {compact_name.replace('_', ' ')}"

    verb_map = {
        "get": "returns",
        "is": "checks whether",
        "set": "sets",
        "create": "creates",
        "destroy": "destroys",
        "free": "frees",
        "init": "initializes",
        "initialize": "initializes",
        "fini": "finalizes",
        "finalize": "finalizes",
        "parse": "parses",
        "format": "formats",
        "compare": "compares",
        "find": "finds",
        "split": "splits",
        "join": "joins",
        "copy": "copies",
        "remove": "removes",
        "add": "adds",
        "clear": "clears",
        "load": "loads",
        "unload": "unloads",
        "open": "opens",
        "close": "closes",
        "write": "writes",
        "read": "reads",
        "allocate": "allocates",
        "deallocate": "deallocates",
        "reallocate": "reallocates",
    }
    tokens = compact_name.split("_") if compact_name else [function_name]
    verb = tokens[0] if tokens else function_name
    remainder = " ".join(tokens[1:]).strip()
    if verb in verb_map:
        return f"{verb_map[verb]} {remainder}".strip()
    return f"operates on {compact_name.replace('_', ' ').strip()}".strip()


def infer_control_flow_elements(function_code: str) -> list[str]:
    elements = [kw for kw in CONTROL_FLOW_KEYWORDS if re.search(rf"\b{kw}\b", function_code)]
    return elements


def infer_side_effects(function_code: str) -> list[str]:
    labels: list[str] = []
    if any(re.search(pattern, function_code) for pattern in IO_PATTERNS):
        labels.append("io")
    if any(re.search(pattern, function_code) for pattern in MEMORY_PATTERNS):
        labels.append("memory")
    if any(re.search(pattern, function_code) for pattern in HARDWARE_PATTERNS):
        labels.append("hardware")
    if not labels:
        labels.append("none")
    return labels


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