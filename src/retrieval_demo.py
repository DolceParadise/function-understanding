#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from embedding_demo_utils import (
    DEFAULT_DATA_GLOB,
    EmbeddedFunction,
    find_query_record,
    load_embedded_functions,
    retrieval_payload,
    top_k_similar,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve similar functions for a query function.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--data-glob", default=DEFAULT_DATA_GLOB)
    parser.add_argument("--query-function", default=None)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = args.repo_root / "data"
    if args.output_dir is None:
        args.output_dir = args.repo_root / "data"
    return args


def build_retrieval_result(
    repo_root: Path,
    data_dir: Path | None,
    data_glob: str,
    query_function: str | None,
    query_index: int,
    top_k: int,
) -> tuple[dict, EmbeddedFunction, list[tuple[float, EmbeddedFunction]]]:
    resolved_data_dir = data_dir or (repo_root / "data")
    records = load_embedded_functions(resolved_data_dir, data_glob)
    query_record = find_query_record(records, query_function, query_index)
    neighbors = top_k_similar(records, query_record, top_k)
    return retrieval_payload(query_record, neighbors), query_record, neighbors


def main() -> None:
    args = parse_args()
    payload, query_record, neighbors = build_retrieval_result(
        args.repo_root,
        args.data_dir,
        args.data_glob,
        args.query_function,
        args.query_index,
        args.top_k,
    )

    write_json(args.output_dir / "function_retrieval.json", payload)

    print("RETRIEVAL BY QUERY FUNCTION")
    print(f"query function: {query_record.function_name}")
    print(f"query purpose: {query_record.label or 'n/a'}")
    print("nearest neighbors:")
    for rank, (score, record) in enumerate(neighbors, start=1):
        print(f"{rank}. {record.function_name} ({score:.4f}) - {record.label or 'n/a'}")


if __name__ == "__main__":
    main()