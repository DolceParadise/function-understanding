#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from embedding_demo_utils import (
    DEFAULT_DATA_GLOB,
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

    clusters = []
    for cluster_index in range(cluster_count):
        members = [record for record, assignment in zip(records, assignments) if assignment == cluster_index]
        label_counts = Counter(record.label or "unlabeled" for record in members)
        dominant_label, dominant_count = ("unlabeled", 0)
        if label_counts:
            dominant_label, dominant_count = label_counts.most_common(1)[0]

        clusters.append(
            {
                "cluster_id": cluster_index,
                "size": len(members),
                "dominant_purpose": dominant_label,
                "dominant_purpose_count": dominant_count,
                "purpose_counts": dict(sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))),
                "functions": [
                    {
                        "function_name": record.function_name,
                        "file_path": record.file_path,
                        "source_file": record.source_file,
                        "label": record.label,
                    }
                    for record in members
                ],
            }
        )

    write_json(
        args.output_dir / "semantic_purpose_clustering.json",
        {
            "cluster_count": cluster_count,
            "clusters": clusters,
        },
    )

    print("CLUSTERING BY SEMANTIC PURPOSE")
    for cluster in clusters:
        print(
            f"cluster {cluster['cluster_id']} ({cluster['size']} functions) - "
            f"dominant purpose: {cluster['dominant_purpose']}"
        )
        for function in cluster["functions"][:5]:
            print(f"  - {function['function_name']} | {function['file_path']} | purpose: {function['label'] or 'n/a'}")
        if cluster["purpose_counts"]:
            top_purposes = ", ".join(
                f"{purpose} ({count})" for purpose, count in list(cluster["purpose_counts"].items())[:3]
            )
            print(f"  purpose mix: {top_purposes}")
        print()


if __name__ == "__main__":
    main()