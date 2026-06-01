#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from embedding_demo_utils import (
    DEFAULT_DATA_GLOB,
    EmbeddedFunction,
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


def build_clustering_result(
    repo_root: Path,
    data_dir: Path | None,
    data_glob: str,
    clusters: int,
    max_iterations: int,
    seed: int,
) -> dict:
    resolved_data_dir = data_dir or (repo_root / "data")
    records = load_embedded_functions(resolved_data_dir, data_glob)
    assignments = kmeans_cluster(records, clusters, max_iterations, seed)
    cluster_count = min(clusters, len(set(assignments)))

    cluster_payloads = []
    for cluster_index in range(cluster_count):
        members = [record for record, assignment in zip(records, assignments) if assignment == cluster_index]
        label_counts = Counter(record.label or "unlabeled" for record in members)
        dominant_label, dominant_count = ("unlabeled", 0)
        if label_counts:
            dominant_label, dominant_count = label_counts.most_common(1)[0]

        cluster_payloads.append(
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

    return {
        "cluster_count": cluster_count,
        "clusters": cluster_payloads,
    }


def main() -> None:
    args = parse_args()
    payload = build_clustering_result(
        args.repo_root,
        args.data_dir,
        args.data_glob,
        args.clusters,
        args.max_iterations,
        args.seed,
    )

    write_json(
        args.output_dir / "semantic_purpose_clustering.json",
        payload,
    )

    print("CLUSTERING BY SEMANTIC PURPOSE")
    for cluster in payload["clusters"]:
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