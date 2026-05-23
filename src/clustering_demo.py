#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from embedding_demo_utils import (
    DEFAULT_DATA_GLOB,
    cluster_payload,
    describe_function,
    kmeans_cluster,
    load_embedded_functions,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write clustering results for saved function embeddings.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--data-glob", default=DEFAULT_DATA_GLOB)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
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
    assignments = kmeans_cluster(records, args.clusters, args.max_iterations, args.seed)
    cluster_count = min(args.clusters, len(set(assignments)))

    write_json(args.output_dir / "semantic_purpose_clustering.json", cluster_payload(records, assignments, cluster_count))

    print("CLUSTERING BY SEMANTIC PURPOSE")
    for cluster_index in range(cluster_count):
        members = [record for record, assignment in zip(records, assignments) if assignment == cluster_index]
        print(f"cluster {cluster_index} ({len(members)} functions)")
        for record in members[:5]:
            print(f"  - {describe_function(record)}")
        print()


if __name__ == "__main__":
    main()