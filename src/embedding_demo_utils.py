from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_GLOB = "*_embeddings.jsonl"


@dataclass(frozen=True)
class EmbeddedFunction:
    function_name: str
    file_path: str
    function_code: str
    label: str
    embedding: list[float]
    source_file: str


def iter_embedding_files(data_dir: Path, data_glob: str = DEFAULT_DATA_GLOB) -> Iterable[Path]:
    for path in sorted(data_dir.rglob(data_glob)):
        if path.is_file():
            yield path


def load_embedded_functions(data_dir: Path, data_glob: str = DEFAULT_DATA_GLOB) -> list[EmbeddedFunction]:
    records: list[EmbeddedFunction] = []
    for source_file in iter_embedding_files(data_dir, data_glob):
        with source_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                embedding = payload.get("embedding")
                if not isinstance(embedding, list) or not embedding:
                    raise ValueError(f"missing embedding in {source_file} at line {line_number}")

                records.append(
                    EmbeddedFunction(
                        function_name=str(payload.get("function_name", "")).strip(),
                        file_path=str(payload.get("file_path", "")).strip(),
                        function_code=str(payload.get("function_code", "")).strip(),
                        label=str(payload.get("labels", {}).get("high_level_purpose", "")).strip(),
                        embedding=[float(value) for value in embedding],
                        source_file=str(source_file),
                    )
                )
    if not records:
        raise SystemExit(f"no embedding records found under {data_dir}")
    return records


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(l * r for l, r in zip(left, right))


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def normalized_cosine(left: list[float], right: list[float]) -> float:
    denominator = vector_norm(left) * vector_norm(right)
    if denominator == 0:
        return 0.0
    return cosine_similarity(left, right) / denominator


def describe_function(record: EmbeddedFunction) -> str:
    label_text = record.label or "n/a"
    return f"{record.function_name} | {record.file_path} | purpose: {label_text}"


def find_query_record(records: list[EmbeddedFunction], query_function: str | None, query_index: int) -> EmbeddedFunction:
    if query_function:
        for record in records:
            if record.function_name == query_function:
                return record
        raise SystemExit(f"function name '{query_function}' was not found in the embedding files")

    if query_index < 0 or query_index >= len(records):
        raise SystemExit(f"query index {query_index} is out of range for {len(records)} records")
    return records[query_index]


def top_k_similar(records: list[EmbeddedFunction], query_record: EmbeddedFunction, top_k: int) -> list[tuple[float, EmbeddedFunction]]:
    scored: list[tuple[float, EmbeddedFunction]] = []
    for record in records:
        if record == query_record:
            continue
        score = normalized_cosine(query_record.embedding, record.embedding)
        scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def initialize_centroids(vectors: list[list[float]], cluster_count: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    indices = list(range(len(vectors)))
    if cluster_count >= len(indices):
        return [vectors[index][:] for index in indices]
    chosen = rng.sample(indices, cluster_count)
    return [vectors[index][:] for index in chosen]


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dimensions = len(vectors[0])
    totals = [0.0] * dimensions
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    count = float(len(vectors))
    return [value / count for value in totals]


def kmeans_cluster(records: list[EmbeddedFunction], cluster_count: int, max_iterations: int, seed: int) -> list[int]:
    vectors = [record.embedding for record in records]
    if cluster_count <= 1 or len(records) <= 1:
        return [0 for _ in records]

    cluster_count = min(cluster_count, len(records))
    centroids = initialize_centroids(vectors, cluster_count, seed)
    assignments = [0] * len(records)

    for _ in range(max_iterations):
        changed = False
        for index, vector in enumerate(vectors):
            best_cluster = 0
            best_score = float("-inf")
            for cluster_index, centroid in enumerate(centroids):
                score = normalized_cosine(vector, centroid)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_index
            if assignments[index] != best_cluster:
                assignments[index] = best_cluster
                changed = True

        new_centroids: list[list[float]] = []
        for cluster_index in range(cluster_count):
            members = [vectors[index] for index, assignment in enumerate(assignments) if assignment == cluster_index]
            if members:
                new_centroids.append(mean_vector(members))
            else:
                new_centroids.append(centroids[cluster_index])

        centroids = new_centroids
        if not changed:
            break

    return assignments


def similarity_payload(query_record: EmbeddedFunction, results: list[tuple[float, EmbeddedFunction]]) -> dict:
    return {
        "query": {
            "function_name": query_record.function_name,
            "file_path": query_record.file_path,
            "source_file": query_record.source_file,
            "label": query_record.label,
        },
        "results": [
            {
                "score": score,
                "function_name": record.function_name,
                "file_path": record.file_path,
                "source_file": record.source_file,
                "label": record.label,
            }
            for score, record in results
        ],
    }


def retrieval_payload(query_record: EmbeddedFunction, results: list[tuple[float, EmbeddedFunction]]) -> dict:
    return {
        "query_function": query_record.function_name,
        "query_file_path": query_record.file_path,
        "query_source_file": query_record.source_file,
        "query_label": query_record.label,
        "neighbors": [
            {
                "score": score,
                "function_name": record.function_name,
                "file_path": record.file_path,
                "source_file": record.source_file,
                "label": record.label,
            }
            for score, record in results
        ],
    }


def cluster_payload(records: list[EmbeddedFunction], assignments: list[int], cluster_count: int) -> dict:
    clusters: dict[int, list[dict]] = {cluster_index: [] for cluster_index in range(cluster_count)}
    for record, cluster_index in zip(records, assignments):
        clusters.setdefault(cluster_index, []).append(
            {
                "function_name": record.function_name,
                "file_path": record.file_path,
                "source_file": record.source_file,
                "label": record.label,
            }
        )
    return {
        "cluster_count": cluster_count,
        "clusters": [
            {
                "cluster_id": cluster_index,
                "size": len(clusters.get(cluster_index, [])),
                "functions": clusters.get(cluster_index, []),
            }
            for cluster_index in range(cluster_count)
        ],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")