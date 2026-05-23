#!/usr/bin/env python3

"""Generate embeddings for function datapoints stored in JSONL files.

The script reads each JSONL file in the repository data directory, builds a
code-focused text representation for every record, and stores the resulting
embeddings next to the source file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


DEFAULT_MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
DEFAULT_INPUT_GLOB = "*.jsonl"
DEFAULT_OUTPUT_SUFFIX = "_embeddings"
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for JSONL function datapoints in the data directory."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing src/ and data/.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory that contains the input JSONL files and receives the embedding outputs.",
    )
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help="Glob used to select input JSONL files under the data directory.",
    )
    parser.add_argument(
        "--output-suffix",
        default=DEFAULT_OUTPUT_SUFFIX,
        help="Suffix inserted before .jsonl for the generated embedding files.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name to load for embeddings.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Maximum sequence length to pass to the embedding model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size to use while encoding records.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files instead of skipping them.",
    )

    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = args.repo_root / "data"
    return args


def iter_input_files(data_dir: Path, input_glob: str) -> Iterable[Path]:
    for path in sorted(data_dir.rglob(input_glob)):
        if not path.is_file():
            continue
        if path.name.endswith("_embeddings.jsonl"):
            continue
        yield path


def load_records(input_path: Path) -> list[dict]:
    records: list[dict] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {input_path} at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object in {input_path} at line {line_number}")
            records.append(record)
    return records


def build_embedding_text(record: dict) -> str:
    function_name = str(record.get("function_name", "")).strip()
    function_code = str(record.get("function_code", "")).strip()
    file_path = str(record.get("file_path", "")).strip()

    parts = []
    if function_name:
        parts.append(f"function_name: {function_name}")
    if file_path:
        parts.append(f"file_path: {file_path}")
    parts.append("function_code:")
    parts.append(function_code)
    return "\n".join(parts).strip()


def load_embedding_model(model_name: str, max_seq_length: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required to generate embeddings. Install it with `python -m pip install sentence-transformers torch`."
        ) from exc

    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.max_seq_length = max_seq_length
    return model


def encode_texts(model, texts: list[str], batch_size: int) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [[float(value) for value in row] for row in embeddings.tolist()]


def output_path_for(input_path: Path, output_suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{output_suffix}{input_path.suffix}")


def write_embeddings(
    input_path: Path,
    output_path: Path,
    records: list[dict],
    embeddings: list[list[float]],
    model_name: str,
) -> None:
    if len(records) != len(embeddings):
        raise ValueError(f"record and embedding counts differ for {input_path}")

    with output_path.open("w", encoding="utf-8") as handle:
        for record, embedding in zip(records, embeddings):
            payload = dict(record)
            payload["embedding_model"] = model_name
            payload["embedding"] = embedding
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    input_files = list(iter_input_files(data_dir, args.input_glob))
    if not input_files:
        raise SystemExit(f"no JSONL inputs found under {data_dir}")

    model = load_embedding_model(args.model_name, args.max_seq_length)

    for input_path in tqdm(input_files, desc="Embedding JSONL files"):
        output_path = output_path_for(input_path, args.output_suffix)
        if output_path.exists() and not args.overwrite:
            print(f"skipping existing {output_path}")
            continue

        records = load_records(input_path)
        texts = [build_embedding_text(record) for record in records]
        embeddings = encode_texts(model, texts, args.batch_size)
        write_embeddings(input_path, output_path, records, embeddings, args.model_name)
        print(f"wrote {len(records)} embeddings to {output_path}")


if __name__ == "__main__":
    main()