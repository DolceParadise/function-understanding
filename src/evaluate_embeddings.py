#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


DEFAULT_RETRIEVAL_FILE = "function_retrieval.json"
DEFAULT_CLUSTERING_FILE = "semantic_purpose_clustering.json"
DEFAULT_TOP_K_VALUES = (1, 3, 5, 10)
DEFAULT_SOFT_THRESHOLD = 0.6
STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "with",
}
WORD_RE = re.compile(r"[a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and clustering demo outputs from saved embedding runs."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--retrieval-file", type=Path, default=None)
    parser.add_argument("--clustering-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k-values", default=",".join(str(value) for value in DEFAULT_TOP_K_VALUES))
    parser.add_argument("--soft-threshold", type=float, default=DEFAULT_SOFT_THRESHOLD)
    parser.add_argument("--max-single-cluster-share", type=float, default=0.5)

    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = args.repo_root / "data"
    if args.retrieval_file is None:
        args.retrieval_file = args.data_dir / DEFAULT_RETRIEVAL_FILE
    if args.clustering_file is None:
        args.clustering_file = args.data_dir / DEFAULT_CLUSTERING_FILE
    if args.output_dir is None:
        args.output_dir = args.data_dir
    return args


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(WORD_RE.findall(value.lower())).strip()


def label_tokens(label: str | None) -> list[str]:
    if not label:
        return []
    return [token for token in WORD_RE.findall(label.lower()) if token not in STOPWORDS]


def label_similarity(left: str | None, right: str | None) -> float:
    left_tokens = set(label_tokens(left))
    right_tokens = set(label_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0

    overlap = len(left_tokens & right_tokens)
    denominator = min(len(left_tokens), len(right_tokens))
    if denominator == 0:
        return 0.0
    return overlap / denominator


def is_semantic_match(left: str | None, right: str | None, threshold: float) -> bool:
    left_text = canonical_text(left)
    right_text = canonical_text(right)
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    if left_text in right_text or right_text in left_text:
        return True
    return label_similarity(left, right) >= threshold


def format_float(value: float) -> str:
    return f"{value:.4f}"


def compute_retrieval_metrics(payload: dict, top_k_values: list[int], soft_threshold: float) -> dict:
    query_label = payload.get("query_label", "")
    query_identity = (
        payload.get("query_function", ""),
        payload.get("query_file_path", ""),
    )
    neighbors = payload.get("neighbors", [])

    annotated_neighbors = []
    first_strict_rank = None
    first_soft_rank = None

    for rank, neighbor in enumerate(neighbors, start=1):
        neighbor_label = str(neighbor.get("label", ""))
        neighbor_identity = (
            neighbor.get("function_name", ""),
            neighbor.get("file_path", ""),
        )
        if neighbor_identity == query_identity:
            strict_hit = False
            soft_hit = False
        else:
            strict_hit = canonical_text(neighbor_label) == canonical_text(query_label) and canonical_text(query_label) != ""
            soft_hit = is_semantic_match(query_label, neighbor_label, soft_threshold)

        if first_strict_rank is None and strict_hit:
            first_strict_rank = rank
        if first_soft_rank is None and soft_hit:
            first_soft_rank = rank

        annotated_neighbors.append(
            {
                "rank": rank,
                "score": neighbor.get("score", 0.0),
                "function_name": neighbor.get("function_name", ""),
                "file_path": neighbor.get("file_path", ""),
                "source_file": neighbor.get("source_file", ""),
                "label": neighbor_label,
                "strict_relevant": strict_hit,
                "soft_relevant": soft_hit,
            }
        )

    def summarize(relevance_field: str) -> dict:
        first_rank = first_strict_rank if relevance_field == "strict_relevant" else first_soft_rank
        mrr = 0.0 if first_rank is None else 1.0 / float(first_rank)
        recall_at_k = {}
        for top_k in top_k_values:
            if top_k <= 0:
                continue
            recall_at_k[top_k] = any(
                neighbor[relevance_field] for neighbor in annotated_neighbors[: min(top_k, len(annotated_neighbors))]
            )
        return {
            "mrr": mrr,
            "first_relevant_rank": first_rank,
            "recall_at_k": recall_at_k,
        }

    strict_summary = summarize("strict_relevant")
    soft_summary = summarize("soft_relevant")

    return {
        "query": {
            "function_name": payload.get("query_function", ""),
            "file_path": payload.get("query_file_path", ""),
            "source_file": payload.get("query_source_file", ""),
            "label": query_label,
        },
        "available_neighbors": len(annotated_neighbors),
        "neighbors": annotated_neighbors,
        "metrics": {
            "strict": strict_summary,
            "soft": soft_summary,
        },
    }


def cluster_label(value: str | None) -> str:
    return canonical_text(value) or "unlabeled"


def compute_cluster_metrics(payload: dict, max_single_cluster_share: float) -> dict:
    clusters = payload.get("clusters", [])
    cluster_summaries = []
    all_members: list[dict] = []
    cluster_sizes: list[int] = []
    empty_clusters: list[int] = []

    for cluster in clusters:
        members = list(cluster.get("functions", []))
        cluster_id = int(cluster.get("cluster_id", len(cluster_summaries)))
        size = len(members)
        cluster_sizes.append(size)
        if size == 0:
            empty_clusters.append(cluster_id)

        label_counts = Counter(cluster_label(member.get("label")) for member in members)
        dominant_label, dominant_count = ("unlabeled", 0)
        if label_counts:
            dominant_label, dominant_count = label_counts.most_common(1)[0]

        purity = 0.0 if size == 0 else dominant_count / float(size)
        cluster_summaries.append(
            {
                "cluster_id": cluster_id,
                "size": size,
                "dominant_label": dominant_label,
                "dominant_label_count": dominant_count,
                "purity": purity,
                "label_counts": dict(sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))),
            }
        )
        all_members.extend(members)

    total_members = len(all_members)
    weighted_purity = 0.0
    macro_purity = 0.0
    if total_members:
        weighted_purity = sum(summary["purity"] * summary["size"] for summary in cluster_summaries) / float(total_members)
    if cluster_summaries:
        macro_purity = sum(summary["purity"] for summary in cluster_summaries) / float(len(cluster_summaries))

    mean_size = sum(cluster_sizes) / float(len(cluster_sizes)) if cluster_sizes else 0.0
    size_variance = (
        sum((size - mean_size) ** 2 for size in cluster_sizes) / float(len(cluster_sizes))
        if cluster_sizes
        else 0.0
    )
    size_stddev = math.sqrt(size_variance)
    largest_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    largest_cluster_share = (largest_cluster_size / float(total_members)) if total_members else 0.0
    singleton_clusters = sum(1 for size in cluster_sizes if size == 1)

    sanity_checks = [
        {
            "name": "no_empty_clusters",
            "passed": len(empty_clusters) == 0,
            "details": {"empty_clusters": empty_clusters},
        },
        {
            "name": "no_overly_dominant_cluster",
            "passed": largest_cluster_share <= max_single_cluster_share,
            "details": {
                "largest_cluster_share": largest_cluster_share,
                "threshold": max_single_cluster_share,
            },
        },
        {
            "name": "no_excess_singletons",
            "passed": singleton_clusters == 0,
            "details": {"singleton_clusters": singleton_clusters},
        },
    ]

    return {
        "cluster_count": len(clusters),
        "total_items": total_members,
        "metrics": {
            "weighted_purity": weighted_purity,
            "macro_purity": macro_purity,
        },
        "size_summary": {
            "min": min(cluster_sizes) if cluster_sizes else 0,
            "max": largest_cluster_size,
            "mean": mean_size,
            "stddev": size_stddev,
            "largest_cluster_share": largest_cluster_share,
            "empty_clusters": empty_clusters,
            "singleton_clusters": singleton_clusters,
        },
        "sanity_checks": sanity_checks,
        "clusters": cluster_summaries,
    }


def render_retrieval_markdown(payload: dict) -> list[str]:
    lines = [
        "## Retrieval Evaluation",
        "",
        f"Query function: {payload['query']['function_name']}",
        f"Query purpose: {payload['query']['label'] or 'n/a'}",
        "",
        "### Metrics",
        "",
        "| Variant | MRR | First relevant rank | Recall@1 | Recall@3 | Recall@5 | Recall@10 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for variant_name in ("strict", "soft"):
        metrics = payload["metrics"][variant_name]
        recall = metrics["recall_at_k"]
        lines.append(
            "| {variant} | {mrr} | {rank} | {r1} | {r3} | {r5} | {r10} |".format(
                variant=variant_name,
                mrr=format_float(metrics["mrr"]),
                rank=metrics["first_relevant_rank"] if metrics["first_relevant_rank"] is not None else "n/a",
                r1="yes" if recall.get(1, False) else "no",
                r3="yes" if recall.get(3, False) else "no",
                r5="yes" if recall.get(5, False) else "no",
                r10="yes" if recall.get(10, False) else "no",
            )
        )

    lines.extend(
        [
            "",
            "### Ranked Neighbors",
            "",
            "| Rank | Score | Function | Purpose | Strict hit | Soft hit |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for neighbor in payload["neighbors"]:
        lines.append(
            "| {rank} | {score} | {function_name} | {label} | {strict} | {soft} |".format(
                rank=neighbor["rank"],
                score=format_float(float(neighbor.get("score", 0.0))),
                function_name=neighbor.get("function_name", ""),
                label=neighbor.get("label", "n/a") or "n/a",
                strict="yes" if neighbor["strict_relevant"] else "no",
                soft="yes" if neighbor["soft_relevant"] else "no",
            )
        )
    return lines


def render_clustering_markdown(payload: dict) -> list[str]:
    lines = [
        "## Clustering Evaluation",
        "",
        f"Cluster count: {payload['cluster_count']}",
        f"Total items: {payload['total_items']}",
        "",
        "### Metrics",
        "",
        f"Weighted purity: {format_float(payload['metrics']['weighted_purity'])}",
        f"Macro purity: {format_float(payload['metrics']['macro_purity'])}",
        "",
        "### Cluster Sanity Checks",
        "",
        "| Check | Passed | Details |",
        "| --- | --- | --- |",
    ]

    for check in payload["sanity_checks"]:
        lines.append(
            f"| {check['name']} | {'yes' if check['passed'] else 'no'} | {json.dumps(check['details'], sort_keys=True)} |"
        )

    lines.extend(
        [
            "",
            "### Cluster Sizes",
            "",
            f"Min: {payload['size_summary']['min']}",
            f"Max: {payload['size_summary']['max']}",
            f"Mean: {format_float(payload['size_summary']['mean'])}",
            f"Stddev: {format_float(payload['size_summary']['stddev'])}",
            f"Largest cluster share: {format_float(payload['size_summary']['largest_cluster_share'])}",
            "",
            "### Per-Cluster Purity",
            "",
            "| Cluster | Size | Dominant label | Purity | Top labels |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for cluster in payload["clusters"]:
        top_labels = ", ".join(
            f"{label} ({count})" for label, count in list(cluster["label_counts"].items())[:3]
        ) or "n/a"
        lines.append(
            "| {cluster_id} | {size} | {dominant_label} | {purity} | {labels} |".format(
                cluster_id=cluster["cluster_id"],
                size=cluster["size"],
                dominant_label=cluster["dominant_label"],
                purity=format_float(cluster["purity"]),
                labels=top_labels,
            )
        )
    return lines


def main() -> None:
    args = parse_args()
    top_k_values = [int(value) for value in args.top_k_values.split(",") if value.strip()]

    retrieval_payload = compute_retrieval_metrics(load_json(args.retrieval_file), top_k_values, args.soft_threshold)
    clustering_payload = compute_cluster_metrics(load_json(args.clustering_file), args.max_single_cluster_share)

    report = {
        "retrieval": retrieval_payload,
        "clustering": clustering_payload,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_output = args.output_dir / "embedding_evaluation_report.json"
    md_output = args.output_dir / "embedding_evaluation_report.md"

    with json_output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    markdown_lines = [
        "# Embedding Evaluation Report",
        "",
        "This report scores the demo output files produced by the retrieval and clustering scripts.",
        "Retrieval uses label agreement as a semantic proxy, with both strict and soft matching variants.",
        "For full Recall@K coverage, the retrieval demo should emit at least K ranked neighbors.",
        "Clustering is summarized with purity and size-based sanity checks.",
        "",
    ]
    markdown_lines.extend(render_retrieval_markdown(retrieval_payload))
    markdown_lines.extend(["", "---", ""])
    markdown_lines.extend(render_clustering_markdown(clustering_payload))

    with md_output.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown_lines).rstrip() + "\n")

    print(f"wrote {json_output}")
    print(f"wrote {md_output}")


if __name__ == "__main__":
    main()