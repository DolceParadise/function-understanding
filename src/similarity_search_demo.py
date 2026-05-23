#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from embedding_demo_utils import (
    DEFAULT_DATA_GLOB,
    describe_function,
    find_query_record,
    load_embedded_functions,
    similarity_payload,
    top_k_similar,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write similarity search results for saved function embeddings.")
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


def main() -> None:
    args = parse_args()
    records = load_embedded_functions(args.data_dir, args.data_glob)
    query_record = find_query_record(records, args.query_function, args.query_index)
    neighbors = top_k_similar(records, query_record, args.top_k)

    write_json(args.output_dir / "function_similarity_search.json", similarity_payload(query_record, neighbors))

    print("SIMILARITY SEARCH")
    print(f"query: {describe_function(query_record)}")
    for rank, (score, record) in enumerate(neighbors, start=1):
        print(f"{rank}. score={score:.4f} :: {describe_function(record)}")


if __name__ == "__main__":
    main()