from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))

from src.clustering_demo import build_clustering_result  
from src.embedding_demo_utils import DEFAULT_DATA_GLOB, kmeans_cluster, load_embedded_functions  
from src.retrieval_demo import build_retrieval_result  


DATA_DIR = REPO_ROOT / "data"
HTML_TEMPLATE_PATH = REPO_ROOT / "configs" / "function_semantics_lab.html"
DEFAULT_CLUSTER_COUNT = 5
DEFAULT_TOP_K = 10
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_SEED = 7
COLOR_PALETTE = ["#0f766e", "#d97706", "#2563eb", "#dc2626", "#7c3aed", "#16a34a", "#0891b2", "#b45309"]

HTML_TEMPLATE = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 1}


def unique_records(records):
    unique: dict[tuple[str, str], Any] = {}
    for record in records:
        unique.setdefault((record.function_name, record.file_path), record)
    return list(unique.values())


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


def cosine_distance(left: list[float], right: list[float]) -> float:
    numerator = sum(l * r for l, r in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    denominator = left_norm * right_norm
    if denominator == 0:
        return 1.0
    return 1.0 - (numerator / denominator)


ALL_RECORDS = load_embedded_functions(DATA_DIR, DEFAULT_DATA_GLOB)
UNIQUE_RECORDS = unique_records(ALL_RECORDS)


def summary(record: Any, score: float | None = None) -> dict[str, Any]:
    payload = {
        "function_name": record.function_name,
        "file_path": record.file_path,
        "source_file": record.source_file,
        "label": record.label,
    }
    if score is not None:
        payload["score"] = round(score, 4)
    return payload


def lexical_score(query: str, record: Any) -> float:
    q = normalize(query)
    if not q:
        return 0.0
    candidate = normalize(f"{record.function_name} {record.label} {record.file_path}")
    q_tokens = tokens(q)
    c_tokens = tokens(candidate)
    ratio = SequenceMatcher(None, q, candidate).ratio()
    token_overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
    exact = 1.0 if q in candidate else 0.0
    return (0.55 * ratio) + (0.3 * token_overlap) + (0.15 * exact)


def suggest_records(query: str, limit: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(UNIQUE_RECORDS, key=lambda record: lexical_score(query, record), reverse=True)
    return [summary(record, lexical_score(query, record)) for record in ranked[:limit] if lexical_score(query, record) > 0]


def resolve_query_record(query: str):
    normalized_query = normalize(query)
    if not normalized_query:
        return UNIQUE_RECORDS[0]

    for record in UNIQUE_RECORDS:
        if normalize(record.function_name) == normalized_query:
            return record

    ranked = sorted(UNIQUE_RECORDS, key=lambda record: lexical_score(query, record), reverse=True)
    return ranked[0] if ranked else UNIQUE_RECORDS[0]


def metrics_for_query(query_label: str, ranked_neighbors) -> dict[str, Any]:
    query_label = normalize(query_label)
    query_tokens = tokens(query_label)

    def first_rank(predicate) -> int | None:
        for index, (_, record) in enumerate(ranked_neighbors, start=1):
            label = normalize(record.label or "")
            if predicate(label):
                return index
        return None

    strict_rank = first_rank(lambda label: bool(query_label) and label == query_label)
    soft_rank = first_rank(lambda label: bool(query_label) and len(query_tokens & tokens(label)) >= 2)

    def metric(rank: int | None) -> dict[str, Any]:
        return {
            "mrr": round(1 / rank, 4) if rank else 0.0,
            "first_relevant_rank": rank,
            "recall_at_k": {str(k): bool(rank and rank <= k) for k in (1, 3, 5, 10)},
        }

    return {"strict": metric(strict_rank), "soft": metric(soft_rank)}


def build_search_payload(query: str) -> dict[str, Any]:
    selected = resolve_query_record(query)
    retrieval_payload, query_record, neighbors = build_retrieval_result(
        REPO_ROOT,
        DATA_DIR,
        DEFAULT_DATA_GLOB,
        selected.function_name,
        0,
        DEFAULT_TOP_K,
    )
    enriched_neighbors = []
    for neighbor_payload, (_, record) in zip(retrieval_payload["neighbors"], neighbors):
        enriched_neighbors.append({**neighbor_payload, "function_code": record.function_code})

    return {
        "typed_query": query,
        "selected_query": {**summary(query_record), "function_code": query_record.function_code},
        "suggestions": suggest_records(query, 3),
        "retrieval": {**retrieval_payload, "neighbors": enriched_neighbors, "metrics": metrics_for_query(query_record.label or "", neighbors)},
    }


def scale_points(records: list[Any]) -> list[dict[str, Any]]:
    points = []
    for index, record in enumerate(records):
        points.append(
            {
                "id": index,
                "function_name": record.function_name,
                "file_path": record.file_path,
                "source_file": record.source_file,
                "label": record.label,
                "function_code": record.function_code,
                "embedding": record.embedding,
            }
        )
    return points


def build_cluster_payload(cluster_count: int) -> dict[str, Any]:
    cluster_count = max(2, min(cluster_count, len(ALL_RECORDS)))
    clusters = build_clustering_result(REPO_ROOT, DATA_DIR, DEFAULT_DATA_GLOB, cluster_count, DEFAULT_MAX_ITERATIONS, DEFAULT_SEED)
    assignments = kmeans_cluster(ALL_RECORDS, cluster_count, DEFAULT_MAX_ITERATIONS, DEFAULT_SEED)
    points = scale_points(ALL_RECORDS)
    cluster_members: dict[int, list[dict[str, Any]]] = {cluster["cluster_id"]: [] for cluster in clusters["clusters"]}
    for point, cluster_id in zip(points, assignments):
        point["cluster_id"] = cluster_id
        point["color"] = COLOR_PALETTE[cluster_id % len(COLOR_PALETTE)]
        cluster_members.setdefault(cluster_id, []).append(point)

    cluster_centroids: dict[int, list[float]] = {}
    for cluster in clusters["clusters"]:
        cluster_id = cluster["cluster_id"]
        members = [point for point in cluster_members.get(cluster_id, []) if point.get("embedding")]
        member_embeddings = [point["embedding"] for point in members]
        centroid = mean_vector(member_embeddings)
        cluster_centroids[cluster_id] = centroid
        ordered_members = sorted(
            members,
            key=lambda point: cosine_distance(point["embedding"], centroid) if centroid else 0.0,
        )
        for member_index, member in enumerate(ordered_members):
            member["cluster_member_rank"] = member_index
            member["distance_to_centroid"] = round(
                cosine_distance(member["embedding"], centroid),
                6,
            ) if centroid else 0.0

        cluster["centroid"] = centroid
        cluster["members"] = [
            {
                "id": member["id"],
                "function_name": member["function_name"],
                "file_path": member["file_path"],
                "source_file": member["source_file"],
                "label": member["label"],
                "cluster_member_rank": member.get("cluster_member_rank", 0),
                "distance_to_centroid": member.get("distance_to_centroid", 0.0),
            }
            for member in ordered_members
        ]

    sizes = [cluster["size"] for cluster in clusters["clusters"]]
    return {
        **clusters,
        "points": points,
        "palette": COLOR_PALETTE,
        "recommended_cluster_count": DEFAULT_CLUSTER_COUNT,
        "summary": {
            "total_items": len(points),
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "mean_size": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        },
    }


def render_page() -> str:
    return HTML_TEMPLATE.replace("__DEFAULT_CLUSTER_COUNT__", str(DEFAULT_CLUSTER_COUNT)).replace("__COLOR_PALETTE__", json.dumps(COLOR_PALETTE))


class AppHandler(BaseHTTPRequestHandler):
    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/":
            self._html(render_page())
            return

        if parsed.path == "/api/config":
            self._json(
                {
                    "default_cluster_count": DEFAULT_CLUSTER_COUNT,
                    "loaded_count": len(ALL_RECORDS),
                    "source_count": len({record.source_file for record in ALL_RECORDS}),
                    "default_query": UNIQUE_RECORDS[0].function_name,
                    "suggestions": suggest_records(UNIQUE_RECORDS[0].function_name, 3),
                }
            )
            return

        if parsed.path == "/api/suggest":
            self._json({"suggestions": suggest_records(query.get("q", [""])[0], 3)})
            return

        if parsed.path == "/api/search":
            self._json(build_search_payload(query.get("q", [""])[0]))
            return

        if parsed.path == "/api/clusters":
            try:
                cluster_count = int(query.get("count", [str(DEFAULT_CLUSTER_COUNT)])[0])
            except ValueError:
                cluster_count = DEFAULT_CLUSTER_COUNT
            self._json(build_cluster_payload(cluster_count))
            return

        self.send_error(404, "Not found")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print(f"Function Semantics Lab running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()