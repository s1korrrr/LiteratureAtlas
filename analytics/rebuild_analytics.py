#!/usr/bin/env python3
"""
Rebuild the local analytics warehouse for LiteratureAtlas.

Pipeline:
- Read *.paper.json and chunks.json produced by the Swift app (Output/papers, Output/chunks).
- Load into DuckDB (atlas.duckdb) with typed tables for papers, embeddings, chunks, claims, methods.
- Compute local analytics (quality/stability/lifecycle/bridges/citations/claims/methods/workflow/hygiene + baseline novelty/centrality/drift/factors) and export JSON + Parquet into Output/analytics/.

Usage:
    python analytics/rebuild_analytics.py [--base|--root /path/to/repo] [--db atlas.duckdb] [--counterfactual-cutoffs 2010 2015 2020]

The script keeps everything local—no network calls. You can iterate in notebooks by opening atlas.duckdb.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import re
import difflib
from dataclasses import dataclass
from typing import List, Dict, Any

try:
    import duckdb  # pip install duckdb>=1.0.0
except Exception as e:
    raise SystemExit(
        "Missing Python dependency 'duckdb'. Install it with:\n"
        "  python -m pip install -r analytics/requirements.txt\n"
        f"Original error: {e}"
    )

try:
    import numpy as np
    import pandas as pd
except Exception as e:
    raise SystemExit(
        "Missing Python dependencies 'numpy'/'pandas'. Install them with:\n"
        "  python -m pip install -r analytics/requirements.txt\n"
        f"Original error: {e}"
    )
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:
    linear_sum_assignment = None
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD, NMF, PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import trustworthiness
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize
except Exception as e:
    raise SystemExit(
        "Missing Python dependency 'scikit-learn'. Install it with:\n"
        "  python -m pip install -r analytics/requirements.txt\n"
        f"Original error: {e}"
    )

# ---------- Paths ----------

def resolve_paths(base_arg: str | None) -> dict[str, pathlib.Path]:
    repo_root = pathlib.Path(base_arg).expanduser().resolve() if base_arg else pathlib.Path(__file__).resolve().parents[1]
    output = repo_root / "Output"
    papers_dir = output / "papers"
    chunks_path = output / "chunks" / "chunks.json"
    analytics_dir = output / "analytics"
    db_path = output / "atlas.duckdb"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    return {
        "repo_root": repo_root,
        "output": output,
        "papers_dir": papers_dir,
        "chunks_path": chunks_path,
        "analytics_dir": analytics_dir,
        "db_path": db_path,
    }

# ---------- Loading ----------

@dataclass
class PaperRow:
    paper_id: str
    title: str
    original_filename: str
    file_path: str
    version: int | None
    year: int | None
    first_read_at: str | None
    ingested_at: str | None
    summary: str
    intro_summary: str | None
    method_summary: str | None
    results_summary: str | None
    cluster_id: int | None
    primary_cluster_k10: int | None
    primary_cluster_k50: int | None
    tags: list[str] | None
    claims: list[dict[str, Any]] | None
    method_pipeline: dict[str, Any] | None
    assumptions: list[str] | None
    page_count: int | None


def load_papers(papers_dir: pathlib.Path) -> tuple[list[PaperRow], list[list[float]]]:
    rows: list[PaperRow] = []
    embeddings: list[list[float]] = []
    for path in sorted(papers_dir.glob("*.paper.json")):
        data = json.loads(path.read_text())
        emb = data.get("embedding") or []
        if not emb:
            continue
        year = data.get("year")
        cluster_id = data.get("clusterIndex")
        # Future multi-scale fields
        primary_k10 = data.get("primary_cluster_k10") or cluster_id
        primary_k50 = data.get("primary_cluster_k50")
        rows.append(
            PaperRow(
                paper_id=data["id"],
                title=data.get("title", ""),
                original_filename=data.get("originalFilename", path.name),
                file_path=data.get("filePath", str(path)),
                version=data.get("version"),
                year=year,
                first_read_at=data.get("firstReadAt"),
                ingested_at=data.get("ingestedAt"),
                summary=data.get("summary", ""),
                intro_summary=data.get("introSummary"),
                method_summary=data.get("methodSummary"),
                results_summary=data.get("resultsSummary"),
                cluster_id=cluster_id,
                primary_cluster_k10=primary_k10,
                primary_cluster_k50=primary_k50,
                tags=data.get("userTags") or data.get("keywords"),
                claims=data.get("claims"),
                method_pipeline=data.get("methodPipeline"),
                assumptions=data.get("assumptions"),
                page_count=data.get("pageCount"),
            )
        )
        embeddings.append(emb)
    return rows, embeddings


def load_chunks(chunks_path: pathlib.Path) -> list[dict[str, Any]]:
    if not chunks_path.exists():
        return []
    data = json.loads(chunks_path.read_text())
    if not isinstance(data, list):
        return []
    return [chunk for chunk in data if chunk.get("embedding")]


def load_user_events(output_root: pathlib.Path) -> list[dict[str, Any]]:
    # Optional newline-delimited JSON at Output/analytics/user_events.jsonl
    events_path = output_root / "analytics" / "user_events.jsonl"
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
            events.append(evt)
        except json.JSONDecodeError:
            continue
    return events


def load_claim_edge_snapshot(output_root: pathlib.Path) -> dict[str, Any] | None:
    """
    Optional artifact written by the Swift app at Output/analytics/claim_edges.json.
    Expected shape:
      {"generatedAt": "...", "edges": [{"srcPaperID": "...", "dstPaperID": "...", "kind": "...", ...}, ...]}
    """
    path = output_root / "analytics" / "claim_edges.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("edges"), list):
        return data
    return None
# ---------- Helpers ----------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "this", "to", "we", "with", "without", "our", "they", "them", "these", "those",
}


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _tokenize(s: str) -> list[str]:
    s = (s or "").lower()
    tokens = re.split(r"[^a-z0-9]+", s)
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]


def _token_set(s: str) -> set[str]:
    return set(_tokenize(s))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return float(inter / union) if union else 0.0


def compute_corpus_version(rows: list[PaperRow]) -> str:
    """
    Match Swift's AppModel.currentCorpusVersion() hash: djb2 over "filePath|v<version>" joined by "||", mod 2^64.
    """
    parts: list[str] = []
    for row in sorted(rows, key=lambda r: (r.file_path or "")):
        v = row.version if isinstance(row.version, int) else 0
        parts.append(f"{row.file_path}|v{v}")
    signature = "||".join(parts)
    h: int = 5381
    mask = (1 << 64) - 1
    for b in signature.encode("utf-8", errors="ignore"):
        h = ((h << 5) + h + b) & mask
    return str(h)


def tag_text_for_row(row: PaperRow) -> str:
    tokens: list[str] = []
    if row.tags:
        tokens.extend(row.tags)
    if row.assumptions:
        tokens.extend(row.assumptions)
    if row.method_pipeline:
        # pull labels from pipeline steps if present
        if isinstance(row.method_pipeline, dict):
            for step in row.method_pipeline.get("steps", []):
                label = step.get("label")
                if isinstance(label, str):
                    tokens.append(label)
    return " ".join(tokens)

# ---------- Embedding transforms ----------

def whiten_embeddings(embeddings: np.ndarray, max_components: int = 128) -> tuple[np.ndarray, PCA]:
    """
    PCA-whiten embeddings to make distances more isotropic.
    Returns (Z, pca_model) where Z has shape (n, k) with k<=max_components.
    """
    if embeddings.size == 0:
        return embeddings, None
    k = min(max_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=k, whiten=True, random_state=0)
    Z = pca.fit_transform(embeddings)
    return Z.astype(np.float32), pca

# ---------- Analytics ----------

def _flatten_galaxy_clusters(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        out.append(node)
        subs = node.get("subclusters")
        if isinstance(subs, list) and subs:
            out.extend(_flatten_galaxy_clusters(subs))
    return out


def load_galaxy_layout(output_root: pathlib.Path, corpus_version: str) -> dict[str, Any] | None:
    """
    Attempts to load Output/clusters/galaxy_<version>.json (or nearest fallback) and returns:
      {"version": <str>, "clusters": [{"cluster_id": int, "x": float, "y": float, "member_paper_ids": [str]}]}
    """
    clusters_dir = output_root / "clusters"
    preferred = clusters_dir / f"galaxy_{corpus_version}.json"
    candidate: pathlib.Path | None = preferred if preferred.exists() else None
    if candidate is None:
        # Fallback: newest galaxy_*.json
        galaxy_files = sorted(clusters_dir.glob("galaxy_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if galaxy_files:
            candidate = galaxy_files[0]
    if candidate is None or not candidate.exists():
        return None

    try:
        data = json.loads(candidate.read_text())
    except Exception:
        return None

    mega = data.get("megaClusters")
    if not isinstance(mega, list):
        return None
    flattened = _flatten_galaxy_clusters(mega)

    clusters: list[dict[str, Any]] = []
    for c in flattened:
        layout = c.get("layoutPosition")
        if not isinstance(layout, dict):
            continue
        x = layout.get("x")
        y = layout.get("y")
        cid = c.get("id")
        members = c.get("memberPaperIDs") or []
        if cid is None or x is None or y is None:
            continue
        if not isinstance(members, list):
            members = []
        clusters.append(
            {
                "cluster_id": int(cid),
                "x": float(x),
                "y": float(y),
                "member_paper_ids": [str(m) for m in members if m],
            }
        )

    return {"version": str(data.get("version") or corpus_version), "clusters": clusters}


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    # Euclidean distance matrix (n,n), symmetric.
    # Uses (a-b)^2 = a^2 + b^2 - 2ab trick.
    if X.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    G = X @ X.T
    sq = np.diag(G).reshape(-1, 1)
    D2 = sq + sq.T - 2 * G
    D2[D2 < 0] = 0
    return np.sqrt(D2, dtype=np.float32)


def _knn_indices_from_distance(D: np.ndarray, k: int) -> list[list[int]]:
    n = D.shape[0]
    if n == 0:
        return []
    k_eff = max(0, min(k, n - 1))
    out: list[list[int]] = []
    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf
        if k_eff == 0:
            out.append([])
            continue
        idx = np.argpartition(row, k_eff)[:k_eff]
        ordered = idx[np.argsort(row[idx])]
        out.append([int(j) for j in ordered])
    return out


def continuity_score(X: np.ndarray, Y: np.ndarray, k: int = 15) -> float:
    """
    Continuity metric for embeddings/projections (0..1), analogous to trustworthiness.
    Uses Euclidean distances in X and Y.
    """
    n = X.shape[0]
    if n <= 1:
        return 1.0
    k_eff = max(1, min(k, n - 1))
    Dx = _pairwise_distances(X)
    Dy = _pairwise_distances(Y)
    # Ranks in Y
    ranks_y = np.argsort(Dy, axis=1)
    rank_pos_y = np.empty_like(ranks_y)
    for i in range(n):
        rank_pos_y[i, ranks_y[i]] = np.arange(n)
    # Neighbors
    nx = [set(row[1 : k_eff + 1].tolist()) for row in np.argsort(Dx, axis=1)]
    ny = [set(row[1 : k_eff + 1].tolist()) for row in ranks_y]
    # Exclusions: neighbors in X not preserved in Y
    total = 0.0
    for i in range(n):
        Ui = nx[i].difference(ny[i])
        for j in Ui:
            total += max(0, int(rank_pos_y[i, j]) - k_eff)
    denom = n * k_eff * (2 * n - 3 * k_eff - 1)
    if denom <= 0:
        return 1.0
    return float(1.0 - (2.0 / denom) * total)


def map_quality_metrics(
    layout: dict[str, Any] | None,
    df_papers: pd.DataFrame,
    Z: np.ndarray,
    k: int = 15,
    max_clusters: int = 1500,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Computes trustworthiness/continuity + local distortion for the current galaxy layout (cluster-level).
    """
    if layout is None or not layout.get("clusters"):
        return {"available": False, "reason": "no galaxy_<version>.json with layoutPosition found"}

    clusters = layout["clusters"]
    # Build cluster centroids in Z space using membership if possible; otherwise fall back to df_papers cluster_id groups.
    paper_idx = {pid: int(i) for i, pid in enumerate(df_papers.paper_id.tolist())}

    items: list[tuple[int, float, float, np.ndarray]] = []
    for c in clusters:
        cid = int(c["cluster_id"])
        members = [m for m in c.get("member_paper_ids", []) if m in paper_idx]
        if members:
            idxs = [paper_idx[m] for m in members]
            centroid = Z[idxs].mean(axis=0)
        else:
            grp = df_papers[df_papers.cluster_id == cid]
            if grp.empty:
                continue
            idxs = grp.index.tolist()
            centroid = Z[idxs].mean(axis=0)
        items.append((cid, float(c["x"]), float(c["y"]), centroid.astype(np.float32)))

    if not items:
        return {"available": False, "reason": "no clusters with members found for map metrics"}

    # Sample if too large
    if len(items) > max_clusters:
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(len(items), size=max_clusters, replace=False)
        items = [items[i] for i in idx]

    cluster_ids = [cid for cid, _, _, _ in items]
    X = np.stack([vec for _, _, _, vec in items], axis=0)
    Y = np.stack([[x, y] for _, x, y, _ in items], axis=0).astype(np.float32)

    # sklearn.manifold.trustworthiness requires n_neighbors < n_samples / 2
    max_tw = max(1, (X.shape[0] // 2) - 1)
    k_eff = max(1, min(k, X.shape[0] - 1, max_tw))
    tw = float(trustworthiness(X, Y, n_neighbors=k_eff))
    cont = float(continuity_score(X, Y, k=k_eff))

    Dx = _pairwise_distances(X)
    Dy = _pairwise_distances(Y)
    nbh_x = _knn_indices_from_distance(Dx, k_eff)
    nbh_y = _knn_indices_from_distance(Dy, k_eff)
    local: list[dict[str, Any]] = []
    overlaps: list[float] = []
    for i, cid in enumerate(cluster_ids):
        sx = set(nbh_x[i])
        sy = set(nbh_y[i])
        if k_eff == 0:
            ov = 1.0
        else:
            ov = len(sx.intersection(sy)) / float(k_eff)
        overlaps.append(ov)
        local.append({"cluster_id": int(cid), "neighbor_overlap": float(ov), "distortion": float(1.0 - ov)})

    return {
        "available": True,
        "galaxy_version": layout.get("version"),
        "level": "cluster",
        "k": int(k_eff),
        "trustworthiness": tw,
        "continuity": cont,
        "avg_neighbor_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "local_distortion": local,
    }


def paper_layout_quality_metrics(
    df_papers: pd.DataFrame,
    Z: np.ndarray,
    k: int = 15,
    max_n: int = 4000,
    seed: int = 0,
    grid_bins: int = 10,
) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Paper-level layout fidelity using a cheap, deterministic 2D projection:
      - Layout: first 2 dims of Z (PCA-whitened space), min-max scaled.
      - Metrics: trustworthiness, continuity (sampled), and per-paper neighbor-overlap distortion
        between kNN in Z (cosine) and kNN in 2D (euclidean).

    Returns (summary_dict, distortion_by_paper_id).
    """
    paper_ids = [str(pid) for pid in df_papers.paper_id.tolist()]
    n = len(paper_ids)
    if n <= 2 or Z.shape[1] < 2:
        return ({"available": False, "reason": "need >=3 papers and dim>=2"}, {})

    # 2D projection (deterministic)
    Y0 = Z[:, :2].astype(np.float32)
    mins = Y0.min(axis=0)
    maxs = Y0.max(axis=0)
    span = maxs - mins
    span[span == 0] = 1.0
    Y = (Y0 - mins) / span

    # Fidelity metrics (sampled if large)
    rng = np.random.default_rng(seed=seed)
    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        Xs = Z[idx]
        Ys = Y[idx]
    else:
        idx = np.arange(n)
        Xs = Z
        Ys = Y

    # trustworthiness requires n_neighbors < n_samples/2
    max_tw = max(1, (Xs.shape[0] // 2) - 1)
    k_eval = max(1, min(k, Xs.shape[0] - 1, max_tw))
    tw = float(trustworthiness(Xs, Ys, n_neighbors=k_eval))
    cont = float(continuity_score(Xs, Ys, k=k_eval))

    # Per-paper neighbor overlap (full set). When querying kneighbors() on the fitted set (X=None),
    # sklearn requires n_neighbors < n_samples_fit, so cap at n-2 (then we add +1 for "self").
    k_eff = max(1, min(k, n - 2))
    nn_high = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine")
    nn_high.fit(Z)
    idx_high = nn_high.kneighbors(return_distance=False)[:, 1:]

    nn_low = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn_low.fit(Y)
    idx_low = nn_low.kneighbors(return_distance=False)[:, 1:]

    distort: dict[str, float] = {}
    overlaps: list[float] = []
    for i, pid in enumerate(paper_ids):
        a = set(int(x) for x in idx_high[i])
        b = set(int(x) for x in idx_low[i])
        ov = len(a.intersection(b)) / float(k_eff) if k_eff > 0 else 1.0
        overlaps.append(ov)
        distort[pid] = float(1.0 - ov)

    # Grid summary (mean distortion per cell)
    bins = max(2, int(grid_bins))
    grid_acc: dict[tuple[int, int], list[float]] = {}
    for i, pid in enumerate(paper_ids):
        x = float(Y[i, 0])
        y = float(Y[i, 1])
        xi = min(bins - 1, max(0, int(x * bins)))
        yi = min(bins - 1, max(0, int(y * bins)))
        grid_acc.setdefault((xi, yi), []).append(distort[pid])
    grid = [
        {"x_bin": int(xi), "y_bin": int(yi), "count": int(len(vals)), "avg_distortion": float(np.mean(vals))}
        for (xi, yi), vals in grid_acc.items()
    ]
    grid.sort(key=lambda r: (-r["avg_distortion"], -r["count"]))

    q = np.quantile(np.array(list(distort.values()), dtype=np.float32), [0.5, 0.9, 0.95, 0.99]).tolist() if distort else [0, 0, 0, 0]
    summary = {
        "available": True,
        "level": "paper",
        "method": "pca2",
        "k": int(k_eff),
        "trustworthiness": tw,
        "continuity": cont,
        "avg_neighbor_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "distortion_quantiles": {"p50": float(q[0]), "p90": float(q[1]), "p95": float(q[2]), "p99": float(q[3])},
        "grid_bins": int(bins),
        "grid": grid[: min(120, len(grid))],
    }
    return summary, distort


def compute_cluster_centroids(df_papers: pd.DataFrame, embeddings: np.ndarray) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    if "cluster_id" not in df_papers.columns:
        return centroids
    for cid, idxs in df_papers[df_papers.cluster_id.notna()].groupby("cluster_id").groups.items():
        vectors = embeddings[list(idxs)]
        if len(vectors) == 0:
            continue
        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[int(cid)] = centroid / norm if norm else centroid
    return centroids


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_multi_novelty(df_papers: pd.DataFrame, Z: np.ndarray, drift_vectors: dict[tuple[int, int], np.ndarray]) -> dict[str, dict[str, float]]:
    """
    Returns map paper_id -> {nov_cluster, nov_global, nov_directional}
    using whitened embeddings Z.
    """
    results: dict[str, dict[str, float]] = {}
    centroids = compute_cluster_centroids(df_papers, Z)
    global_mean = Z.mean(axis=0, keepdims=True)

    # Precompute cluster-year centroids for directional component
    cluster_year_centroid: dict[tuple[int, int], np.ndarray] = {}
    for (cid, year), group in df_papers.dropna(subset=["cluster_id", "year"]).groupby(["cluster_id", "year"]):
        idxs = group.index.tolist()
        vecs = Z[idxs]
        if len(vecs) == 0:
            continue
        centroid = vecs.mean(axis=0)
        cluster_year_centroid[(int(cid), int(year))] = centroid

    for i, row in df_papers.iterrows():
        pid = row.paper_id
        emb = Z[i]
        cid = row.cluster_id
        year = row.year
        if cid is None or cid not in centroids:
            continue
        cvec = centroids[int(cid)]
        nov_cluster = float(np.linalg.norm(emb - cvec))
        nov_global = float(np.linalg.norm(emb - global_mean))

        nov_dir = 0.0
        if year is not None:
            key = (int(cid), int(year))
            if key in cluster_year_centroid and (key in drift_vectors):
                drift_vec = drift_vectors[key]
                if np.linalg.norm(drift_vec) > 0:
                    v = emb - cluster_year_centroid[key]
                    denom = np.linalg.norm(v) * np.linalg.norm(drift_vec)
                    if denom > 0:
                        nov_dir = float(np.dot(v, drift_vec) / denom)

        results[pid] = {
            "nov_cluster": nov_cluster,
            "nov_global": nov_global,
            "nov_directional": nov_dir,
        }
    return results


def novelty_uncertainty(df_papers: pd.DataFrame, Z: np.ndarray, centroids: dict[int, np.ndarray], trials: int = 5, noise: float = 0.01) -> dict[str, float]:
    """
    Bootstrap novelty by jittering embeddings.
    Returns std dev of cluster-norm distance across jitters.
    """
    rng = np.random.default_rng(seed=0)
    base = {row.paper_id: float(np.linalg.norm(Z[i] - centroids[row.cluster_id])) if row.cluster_id in centroids else 0.0 for i, row in df_papers.iterrows()}
    accum: dict[str, list[float]] = {pid: [val] for pid, val in base.items()}
    for _ in range(trials):
        jitter = Z + rng.normal(scale=noise, size=Z.shape)
        cent = compute_cluster_centroids(df_papers, jitter)
        for i, row in df_papers.iterrows():
            pid = row.paper_id
            if row.cluster_id not in cent:
                continue
            val = float(np.linalg.norm(jitter[i] - cent[row.cluster_id]))
            accum[pid].append(val)
    stds = {pid: float(np.std(vals)) for pid, vals in accum.items()}
    return stds


def compute_topic_trends(df_papers: pd.DataFrame) -> list[dict[str, Any]]:
    subset = df_papers.dropna(subset=["year", "cluster_id"])
    if subset.empty:
        return []
    grouped = (
        subset.groupby(["cluster_id", "year"]).size().reset_index(name="count").sort_values(["cluster_id", "year"])
    )
    return [
        {
            "cluster_id": int(row.cluster_id),
            "year": int(row.year),
            "count": int(row["count"]),
        }
        for _, row in grouped.iterrows()
    ]


# ---------- Factor model (lightweight PCA surrogate) ----------

def compute_factor_loadings(
    embeddings: np.ndarray,
    paper_ids: list[str],
    tag_texts: list[str],
    n_factors: int = 8,
):
    """
    Build a hybrid factor model:
    - dense semantic factors from embeddings (PCA via TruncatedSVD)
    - interpretable tags via TF-IDF + NMF to derive human-readable factor names
    """
    if len(paper_ids) == 0:
        return [], [], []

    # ----- Dense semantic components (PCA/SVD) -----
    X_emb = embeddings - embeddings.mean(axis=0, keepdims=True)
    svd = TruncatedSVD(n_components=min(n_factors, X_emb.shape[1] - 1), random_state=0)
    dense_scores = svd.fit_transform(X_emb)
    dense_comps = svd.components_

    # ----- Tag-driven components (TF-IDF + NMF for labels) -----
    tag_texts = [t if isinstance(t, str) else "" for t in tag_texts]
    tfidf = TfidfVectorizer(max_features=600, ngram_range=(1, 2), min_df=1)
    tfidf_mat = tfidf.fit_transform(tag_texts)
    vocab = np.array(tfidf.get_feature_names_out())

    # If tags are too sparse, fall back to dense labels
    nmf_components = []
    factor_labels: list[str] = []
    if tfidf_mat.shape[0] >= 3 and tfidf_mat.shape[1] >= 4:
        nmf = NMF(n_components=min(n_factors, tfidf_mat.shape[1], tfidf_mat.shape[0]), init="nndsvda", random_state=0, max_iter=300)
        W = nmf.fit_transform(tfidf_mat)
        H = nmf.components_
        nmf_components = H
        for row in H:
            top_idx = row.argsort()[-4:][::-1]
            labels = [vocab[i] for i in top_idx if row[i] > 0]
            factor_labels.append(", ".join(labels) if labels else "Factor")
    else:
        factor_labels = [f"Factor {i+1}" for i in range(n_factors)]

    # ----- Combine scores (semantic) and return -----
    loadings = []
    for pid, dense_row in zip(paper_ids, dense_scores):
        loadings.append({"paper_id": pid, "scores": dense_row.tolist()})

    factors = dense_comps.tolist()
    # If we have nmf components, blend names; otherwise fallback
    if len(factor_labels) < n_factors:
        factor_labels += [f"Factor {i+1}" for i in range(len(factor_labels), n_factors)]

    return factors, loadings, factor_labels[:n_factors]


def factor_exposures_over_time(loadings: list[dict[str, Any]], years: list[int | None], n_factors: int) -> list[dict[str, Any]]:
    rows = []
    for item, year in zip(loadings, years):
        if year is None or (isinstance(year, float) and math.isnan(year)):
            continue
        f_max = min(n_factors, len(item["scores"]))
        for f in range(f_max):
            rows.append({"year": int(year), "factor": f, "score": float(item["scores"][f])})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    agg = df.groupby(["year", "factor"]).score.mean().reset_index()
    return [
        {"year": int(r.year), "factor": int(r.factor), "score": float(r.score)}
        for _, r in agg.iterrows()
    ]


def factor_exposures_from_reads(loadings: list[dict[str, Any]], df_papers: pd.DataFrame, user_events: list[dict[str, Any]], n_factors: int) -> list[dict[str, Any]]:
    """
    Build exposures based on user interaction/reading time rather than publication year.
    Priority:
      1) earliest user event per paper (timestamp year)
      2) paper.first_read_at year
      3) ingest year (ingested_at)
      4) fallback to publication year
    """
    # Build earliest event year per paper_id
    event_year: Dict[str, int] = {}
    for evt in user_events:
        pid = evt.get("paper_id")
        ts = evt.get("timestamp")
        if not pid or not ts or not isinstance(ts, str):
            continue
        try:
            year = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).year
        except Exception:
            continue
        if pid not in event_year or year < event_year[pid]:
            event_year[pid] = year

    rows: list[dict[str, Any]] = []
    for item in loadings:
        pid = item["paper_id"]
        # Find source row
        row = df_papers[df_papers.paper_id == pid]
        if row.empty:
            continue
        yr: int | None = None
        if pid in event_year:
            yr = event_year[pid]
        else:
            first_read = row.iloc[0].get("first_read_at")
            if isinstance(first_read, str):
                try:
                    yr = dt.datetime.fromisoformat(first_read.replace("Z", "+00:00")).year
                except Exception:
                    pass
            if yr is None:
                ing = row.iloc[0].get("ingested_at")
                if isinstance(ing, str):
                    try:
                        yr = dt.datetime.fromisoformat(ing.replace("Z", "+00:00")).year
                    except Exception:
                        pass
            if yr is None:
                pub = row.iloc[0].get("year")
                if not (isinstance(pub, float) and math.isnan(pub)):
                    yr = int(pub) if pub is not None else None
        if yr is None:
            continue
        for f_idx, score in enumerate(item["scores"][:n_factors]):
            rows.append({"year": int(yr), "factor": int(f_idx), "score": float(score)})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    agg = df.groupby(["year", "factor"]).score.mean().reset_index()
    return [
        {"year": int(r.year), "factor": int(r.factor), "score": float(r.score)}
        for _, r in agg.iterrows()
    ]


def build_knn(embeddings: np.ndarray, paper_ids: list[str], k: int = 8) -> list[dict[str, Any]]:
    # Normalize to unit vectors for cosine similarity.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = embeddings / norms
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)

    n = len(paper_ids)
    k_eff = max(0, min(k, n - 1))

    neighbors: list[dict[str, Any]] = []
    for i, pid in enumerate(paper_ids):
        row = sims[i]
        if k_eff == 0:
            top = []
        else:
            idx = np.argpartition(-row, k_eff)[:k_eff]
            ordered = idx[np.argsort(-row[idx])]
            top = [
                {"paper_id": paper_ids[j], "score": float(row[j])}
                for j in ordered
                if not math.isnan(row[j])
            ]
        neighbors.append({"paper_id": pid, "neighbors": top})
    return neighbors


def cluster_stability_and_boundary(
    df_papers: pd.DataFrame,
    Z: np.ndarray,
    trials: int = 20,
    noise: float = 0.01,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Estimates assignment stability to the *existing* cluster_id labels under embedding jitter.
    Returns:
      {
        "trials": int,
        "noise": float,
        "clusters": [cluster_id...],
        "per_paper": [{paper_id, cluster_confidence, ambiguity, top1_cluster, top2_cluster, margin, dist_margin}],
        "per_cluster": [{cluster_id, cohesion, avg_confidence, avg_ambiguity, size}],
      }
    """
    subset = df_papers[df_papers.cluster_id.notna()].copy()
    if subset.empty:
        return {"available": False, "reason": "no cluster_id labels present"}

    # Ensure deterministic ordering aligned with Z rows
    idxs = subset.index.tolist()
    Zs = Z[idxs]
    paper_ids = subset.paper_id.tolist()
    cluster_ids = sorted({int(c) for c in subset.cluster_id.tolist() if not pd.isna(c)})
    k = len(cluster_ids)
    if k <= 1:
        return {"available": False, "reason": "need >=2 clusters for stability metrics"}

    cid_to_col = {cid: i for i, cid in enumerate(cluster_ids)}
    base_labels = np.array([cid_to_col[int(c)] for c in subset.cluster_id.tolist()], dtype=np.int32)

    # Baseline centroids using current assignments.
    centroids = np.zeros((k, Zs.shape[1]), dtype=np.float32)
    counts_c = np.zeros(k, dtype=np.int32)
    for i, lab in enumerate(base_labels):
        centroids[lab] += Zs[i]
        counts_c[lab] += 1
    for j in range(k):
        if counts_c[j] > 0:
            centroids[j] /= float(counts_c[j])

    # Baseline distance margins
    d0 = np.linalg.norm(Zs[:, None, :] - centroids[None, :, :], axis=2)
    order = np.argsort(d0, axis=1)
    best = d0[np.arange(d0.shape[0]), order[:, 0]]
    second = d0[np.arange(d0.shape[0]), order[:, 1]]
    dist_margin = (second - best).astype(np.float32)  # larger => clearer assignment

    # Jitter trials: recompute centroids from jittered vectors (same base_labels) then reassign to nearest centroid.
    rng = np.random.default_rng(seed=seed)
    counts = np.zeros((len(paper_ids), k), dtype=np.int32)
    counts[np.arange(len(paper_ids)), base_labels] += 1

    for _ in range(max(0, trials - 1)):
        jitter = Zs + rng.normal(scale=noise, size=Zs.shape).astype(np.float32)
        cent = np.zeros_like(centroids)
        for i, lab in enumerate(base_labels):
            cent[lab] += jitter[i]
        for j in range(k):
            if counts_c[j] > 0:
                cent[j] /= float(counts_c[j])
        d = np.linalg.norm(jitter[:, None, :] - cent[None, :, :], axis=2)
        labs = np.argmin(d, axis=1)
        counts[np.arange(len(paper_ids)), labs] += 1

    p = counts / float(trials)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)
    ambiguity = (ent / float(np.log(k))).astype(np.float32)
    conf = p.max(axis=1).astype(np.float32)
    top1 = np.argmax(p, axis=1)
    top2 = np.argsort(-p, axis=1)[:, 1]
    margin = (p[np.arange(len(paper_ids)), top1] - p[np.arange(len(paper_ids)), top2]).astype(np.float32)

    per_paper: list[dict[str, Any]] = []
    for pid, t1, t2, c, a, m, dm in zip(paper_ids, top1, top2, conf, ambiguity, margin, dist_margin):
        per_paper.append(
            {
                "paper_id": pid,
                "cluster_confidence": float(c),
                "ambiguity": float(a),
                "top1_cluster": int(cluster_ids[int(t1)]),
                "top2_cluster": int(cluster_ids[int(t2)]),
                "margin": float(m),
                "dist_margin": float(dm),
            }
        )

    # Cohesion per cluster (cosine to centroid)
    norms = np.linalg.norm(Zs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Zs_n = Zs / norms
    cent_n = centroids.copy()
    c_norm = np.linalg.norm(cent_n, axis=1, keepdims=True)
    c_norm[c_norm == 0] = 1.0
    cent_n = cent_n / c_norm
    sims = (Zs_n * cent_n[base_labels]).sum(axis=1)

    per_cluster: list[dict[str, Any]] = []
    for cid in cluster_ids:
        col = cid_to_col[cid]
        mask = base_labels == col
        size = int(mask.sum())
        if size == 0:
            continue
        per_cluster.append(
            {
                "cluster_id": int(cid),
                "size": size,
                "cohesion": float(np.mean(sims[mask])),
                "avg_confidence": float(np.mean(conf[mask])),
                "avg_ambiguity": float(np.mean(ambiguity[mask])),
            }
        )
    per_cluster.sort(key=lambda x: x["cohesion"], reverse=True)

    return {
        "available": True,
        "trials": int(trials),
        "noise": float(noise),
        "clusters": cluster_ids,
        "per_paper": per_paper,
        "per_cluster": per_cluster,
    }


def _align_clusters_by_centroid(base_centroids: np.ndarray, run_centroids: np.ndarray) -> dict[int, int]:
    """
    Align run cluster indices -> base cluster indices by minimizing Euclidean distance between centroids.
    Falls back to greedy matching if SciPy is unavailable.
    """
    if base_centroids.size == 0 or run_centroids.size == 0:
        return {}
    A = base_centroids.astype(np.float32)
    B = run_centroids.astype(np.float32)
    # Cost matrix (k,k) where lower is better.
    cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)

    k = cost.shape[0]
    if linear_sum_assignment is not None and cost.shape[0] == cost.shape[1]:
        row_ind, col_ind = linear_sum_assignment(cost)
        return {int(col): int(row) for row, col in zip(row_ind, col_ind)}

    # Greedy: repeatedly pick best remaining pair.
    mapping: dict[int, int] = {}
    used_base: set[int] = set()
    used_run: set[int] = set()
    flat = [(float(cost[i, j]), int(i), int(j)) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    flat.sort(key=lambda x: x[0])
    for _, i, j in flat:
        if i in used_base or j in used_run:
            continue
        mapping[j] = i
        used_base.add(i)
        used_run.add(j)
        if len(mapping) >= k:
            break
    # Fill any gaps
    for j in range(cost.shape[1]):
        if j in mapping:
            continue
        for i in range(cost.shape[0]):
            if i not in used_base:
                mapping[j] = i
                used_base.add(i)
                break
    return mapping


def cluster_stability_multi_seed_kmeans(
    df_papers: pd.DataFrame,
    Z: np.ndarray,
    runs: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Estimates assignment stability across KMeans reruns (different seeds) while aligning labels to the existing
    df_papers.cluster_id via centroid matching.

    Top-level keys match AnalyticsSummary.StabilitySection for easy decoding.
    """
    subset = df_papers[df_papers.cluster_id.notna()].copy()
    if subset.empty:
        return {"available": False, "reason": "no cluster_id labels present"}

    idxs = subset.index.tolist()
    Zs = Z[idxs].astype(np.float32)
    paper_ids = subset.paper_id.tolist()
    cluster_ids = sorted({int(c) for c in subset.cluster_id.tolist() if not pd.isna(c)})
    k = len(cluster_ids)
    if k <= 1:
        return {"available": False, "reason": "need >=2 clusters for stability metrics"}
    runs_eff = max(1, int(runs))

    cid_to_col = {cid: i for i, cid in enumerate(cluster_ids)}
    base_labels = np.array([cid_to_col[int(c)] for c in subset.cluster_id.tolist()], dtype=np.int32)

    # Base centroids (canonical clusters)
    base_centroids = np.zeros((k, Zs.shape[1]), dtype=np.float32)
    counts_c = np.zeros(k, dtype=np.int32)
    for i, lab in enumerate(base_labels):
        base_centroids[lab] += Zs[i]
        counts_c[lab] += 1
    for j in range(k):
        if counts_c[j] > 0:
            base_centroids[j] /= float(counts_c[j])

    # Distance margin to 2nd closest canonical centroid
    d0 = np.linalg.norm(Zs[:, None, :] - base_centroids[None, :, :], axis=2)
    order = np.argsort(d0, axis=1)
    best = d0[np.arange(d0.shape[0]), order[:, 0]]
    second = d0[np.arange(d0.shape[0]), order[:, 1]]
    dist_margin = (second - best).astype(np.float32)

    counts = np.zeros((len(paper_ids), k), dtype=np.int32)
    # Run KMeans multiple times; align to base centroids
    for r in range(runs_eff):
        km = KMeans(n_clusters=k, n_init=1, random_state=seed + r, algorithm="lloyd")
        labels = km.fit_predict(Zs)
        run_centroids = km.cluster_centers_.astype(np.float32)
        mapping = _align_clusters_by_centroid(base_centroids, run_centroids)
        aligned = np.array([mapping.get(int(l), int(l)) for l in labels], dtype=np.int32)
        counts[np.arange(len(paper_ids)), aligned] += 1

    p = counts / float(runs_eff)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)
    ambiguity = (ent / float(np.log(k))).astype(np.float32)

    top1 = np.argmax(p, axis=1)
    top2 = np.argsort(-p, axis=1)[:, 1]
    best_conf = p[np.arange(len(paper_ids)), top1].astype(np.float32)
    margin = (p[np.arange(len(paper_ids)), top1] - p[np.arange(len(paper_ids)), top2]).astype(np.float32)
    agreement = p[np.arange(len(paper_ids)), base_labels].astype(np.float32)

    per_paper: list[dict[str, Any]] = []
    for pid, t1, t2, agree, bc, a, m, dm in zip(paper_ids, top1, top2, agreement, best_conf, ambiguity, margin, dist_margin):
        per_paper.append(
            {
                "paper_id": pid,
                # Confidence is agreement with the current stored assignment (cluster_id)
                "cluster_confidence": float(agree),
                "best_confidence": float(bc),
                "ambiguity": float(a),
                "top1_cluster": int(cluster_ids[int(t1)]),
                "top2_cluster": int(cluster_ids[int(t2)]),
                "margin": float(m),
                "dist_margin": float(dm),
            }
        )

    # Cohesion per cluster (cosine to canonical centroid)
    norms = np.linalg.norm(Zs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Zs_n = Zs / norms
    cent_n = base_centroids.copy()
    c_norm = np.linalg.norm(cent_n, axis=1, keepdims=True)
    c_norm[c_norm == 0] = 1.0
    cent_n = cent_n / c_norm
    sims = (Zs_n * cent_n[base_labels]).sum(axis=1)

    per_cluster: list[dict[str, Any]] = []
    for cid in cluster_ids:
        col = cid_to_col[cid]
        mask = base_labels == col
        size = int(mask.sum())
        if size == 0:
            continue
        per_cluster.append(
            {
                "cluster_id": int(cid),
                "size": size,
                "cohesion": float(np.mean(sims[mask])),
                "avg_confidence": float(np.mean(agreement[mask])),
                "avg_best_confidence": float(np.mean(best_conf[mask])),
                "avg_ambiguity": float(np.mean(ambiguity[mask])),
            }
        )
    per_cluster.sort(key=lambda x: x["cohesion"], reverse=True)

    return {
        "available": True,
        "method": "kmeans_multi_seed",
        "trials": int(runs_eff),
        "noise": None,
        "clusters": cluster_ids,
        "per_paper": per_paper,
        "per_cluster": per_cluster,
    }


def combinational_novelty(df_papers: pd.DataFrame) -> dict[str, float]:
    """
    Compute how unlikely each paper's tag combination is relative to marginal tag frequencies.
    Tags sourced from tags/assumptions/pipeline labels.
    """
    tag_sets: list[set[str]] = []
    for _, row in df_papers.iterrows():
        tokens: set[str] = set()
        tags = row.get("tags") if isinstance(row.get("tags"), list) else []
        tokens.update([t.lower() for t in tags])
        assumptions = row.get("assumptions") if isinstance(row.get("assumptions"), list) else []
        tokens.update([a.lower() for a in assumptions])
        pipeline = row.get("method_pipeline")
        if isinstance(pipeline, dict):
            for step in pipeline.get("steps", []):
                label = step.get("label")
                if isinstance(label, str):
                    tokens.add(label.lower())
        tag_sets.append(tokens)

    vocab: list[str] = sorted({t for s in tag_sets for t in s})
    if not vocab:
        return {row.paper_id: 0.0 for _, row in df_papers.iterrows()}
    idx = {t: i for i, t in enumerate(vocab)}
    n = len(tag_sets)
    freq = np.zeros(len(vocab), dtype=np.float64)
    pair_freq = np.zeros((len(vocab), len(vocab)), dtype=np.float64)

    for s in tag_sets:
        for t in s:
            freq[idx[t]] += 1
        for a in s:
            for b in s:
                if a == b:
                    continue
                pair_freq[idx[a], idx[b]] += 1

    p = (freq + 1) / (n + 2)  # Laplace smoothing
    p_pair = (pair_freq + 1) / (n + 2)

    scores: dict[str, float] = {}
    eps = 1e-9
    for (i, row), tags in zip(df_papers.iterrows(), tag_sets):
        if not tags:
            scores[row.paper_id] = 0.0
            continue
        log_p0 = 0.0
        for t in tags:
            log_p0 += math.log(p[idx[t]])
        # empirical combo via average pairwise probability
        pairs = [(a, b) for a in tags for b in tags if a != b]
        if pairs:
            vals = [p_pair[idx[a], idx[b]] for a, b in pairs]
            p_emp = sum(vals) / len(vals)
        else:
            p_emp = min(p[idx[t]] for t in tags)
        n_comb = -math.log((p_emp + eps) / (math.exp(log_p0) + eps))
        scores[row.paper_id] = float(n_comb)
    return scores


def centrality_from_knn(knn: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in knn:
        neigh = item["neighbors"]
        if not neigh:
            continue
        scores = [n["score"] for n in neigh]
        degree = float(sum(scores))
        avg_sim = float(sum(scores) / len(scores))
        results.append(
            {
                "paper_id": item["paper_id"],
                "weighted_degree": round(degree, 6),
                "average_similarity": round(avg_sim, 6),
                "neighbors": neigh,
            }
        )
    # Rank by degree descending
    results.sort(key=lambda x: x["weighted_degree"], reverse=True)
    return results


def eigen_centrality_from_knn(knn: list[dict[str, Any]], max_iter: int = 80, tol: float = 1e-6) -> dict[str, float]:
    if not knn:
        return {}
    ids = [item["paper_id"] for item in knn]
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    n = len(ids)
    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    for item in knn:
        i = id_to_idx[item["paper_id"]]
        for neigh in item["neighbors"]:
            j = id_to_idx.get(neigh["paper_id"])
            if j is None:
                continue
            row_idx.append(i)
            col_idx.append(j)
            data.append(max(0.0, float(neigh["score"])))
    if not data:
        return {pid: 0.0 for pid in ids}
    # Build sparse-like via lists
    vec = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(max_iter):
        new = np.zeros(n, dtype=np.float64)
        for r, c, w in zip(row_idx, col_idx, data):
            new[r] += w * vec[c]
        norm = np.linalg.norm(new)
        if norm == 0:
            break
        new /= norm
        if np.linalg.norm(new - vec) < tol:
            vec = new
            break
        vec = new
    return {pid: float(vec[id_to_idx[pid]]) for pid in ids}


# ---------- Drift over time ----------

def compute_drift(df_papers: pd.DataFrame, embeddings: np.ndarray) -> list[dict[str, Any]]:
    if "cluster_id" not in df_papers.columns or "year" not in df_papers.columns:
        return []
    data = []
    for cid, group in df_papers.dropna(subset=["cluster_id", "year"]).groupby("cluster_id"):
        years = sorted(group.year.dropna().unique())
        if len(years) < 2:
            continue
        prev_centroid = None
        prev_year = None
        for year in years:
            idxs = group.index[group.year == year].tolist()
            vecs = embeddings[idxs]
            centroid = vecs.mean(axis=0)
            drift_mag = 0.0
            dx = dy = 0.0
            if prev_centroid is not None:
                delta = centroid - prev_centroid
                drift_mag = float(np.linalg.norm(delta))
                if drift_mag > 0:
                    # store first two dims for direction arrow (approx only)
                    if delta.shape[0] >= 2:
                        dx = float(delta[0] / drift_mag)
                        dy = float(delta[1] / drift_mag)
            data.append({"cluster_id": int(cid), "year": int(year), "drift": drift_mag, "dx": dx, "dy": dy, "from_year": prev_year, "to_year": int(year)})
            prev_centroid = centroid
            prev_year = int(year)
    return data


def drift_vector_map(df_papers: pd.DataFrame, embeddings: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """
    Returns dict keyed by (cluster_id, year) -> drift vector from year to next year (whitened space).
    """
    vectors: dict[tuple[int, int], np.ndarray] = {}
    for cid, group in df_papers.dropna(subset=["cluster_id", "year"]).groupby("cluster_id"):
        years = sorted(group.year.dropna().unique())
        if len(years) < 2:
            continue
        for idx in range(len(years) - 1):
            y = int(years[idx])
            y_next = int(years[idx + 1])
            idxs_y = group.index[group.year == y].tolist()
            idxs_next = group.index[group.year == y_next].tolist()
            c_y = embeddings[idxs_y].mean(axis=0)
            c_next = embeddings[idxs_next].mean(axis=0)
            vectors[(int(cid), y)] = c_next - c_y
    return vectors


def drift_contribution(df_papers: pd.DataFrame, Z: np.ndarray, drift_vectors: dict[tuple[int, int], np.ndarray], influence_map: dict[str, float]) -> dict[str, float]:
    """
    Approximate how much each paper pushes its cluster in the observed drift direction.
    """
    contrib: dict[str, float] = {}
    # centroids per cluster-year
    cent: dict[tuple[int, int], np.ndarray] = {}
    for (cid, year), group in df_papers.dropna(subset=["cluster_id", "year"]).groupby(["cluster_id", "year"]):
        idxs = group.index.tolist()
        cent[(int(cid), int(year))] = Z[idxs].mean(axis=0)

    for i, row in df_papers.iterrows():
        pid = row.paper_id
        cid = row.cluster_id
        year = row.year
        if cid is None or year is None:
            contrib[pid] = 0.0
            continue
        key = (int(cid), int(year))
        if key not in drift_vectors or key not in cent:
            contrib[pid] = 0.0
            continue
        v = Z[i] - cent[key]
        drift_vec = drift_vectors[key]
        norm_v = np.linalg.norm(v)
        norm_d = np.linalg.norm(drift_vec)
        if norm_v == 0 or norm_d == 0:
            contrib[pid] = 0.0
            continue
        align = float(np.dot(v, drift_vec) / (norm_v * norm_d))
        attention = influence_map.get(pid, 1.0)
        contrib[pid] = align * norm_v * attention
    return contrib


def drift_volatility(drift_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cluster: Dict[int, list[float]] = {}
    for d in drift_entries:
        cid = d.get("cluster_id")
        if cid is None:
            continue
        by_cluster.setdefault(int(cid), []).append(float(d.get("drift", 0.0)))
    out: list[dict[str, Any]] = []
    for cid, vals in by_cluster.items():
        if not vals:
            continue
        out.append({"cluster_id": cid, "volatility": float(np.std(vals)), "avg_step": float(np.mean(vals))})
    return out


def zscore_by_year(values: dict[str, float], years: dict[str, int]) -> dict[str, float]:
    grouped: dict[int, list[float]] = {}
    for pid, val in values.items():
        y = years.get(pid)
        if y is None:
            continue
        if isinstance(y, float) and math.isnan(y):
            continue
        grouped.setdefault(int(y), []).append(val)
    stats: dict[int, tuple[float, float]] = {}
    for y, vals in grouped.items():
        mu = float(np.mean(vals))
        sigma = float(np.std(vals)) if np.std(vals) > 1e-6 else 1.0
        stats[y] = (mu, sigma)
    z: dict[str, float] = {}
    for pid, val in values.items():
        y = years.get(pid)
        if y is None or math.isnan(y) or y not in stats:
            z[pid] = 0.0
            continue
        mu, sigma = stats[int(y)]
        z[pid] = float((val - mu) / sigma)
    return z


def zscore(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=np.float64)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr)) if np.std(arr) > 1e-6 else 1.0
    return {k: float((v - mu) / sigma) for k, v in values.items()}


# ---------- Influence (simple PageRank) ----------

def pagerank(n: int, edges: list[tuple[int, int]], weights: list[float] | None = None, d: float = 0.85, iters: int = 60):
    if n == 0:
        return []
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    out_w = np.zeros(n, dtype=np.float64)
    if weights is None:
        weights = [1.0] * len(edges)
    for (u, _), w in zip(edges, weights):
        out_w[u] += w
    for _ in range(iters):
        new = np.full(n, (1 - d) / n, dtype=np.float64)
        for (u, v), w in zip(edges, weights):
            if out_w[u] > 0:
                new[v] += d * rank[u] * (w / out_w[u])
        rank = new
    return rank.tolist()


# ---------- Topic lifecycle (bursts / changepoints) ----------

def burst_scores(series: pd.Series, window: int = 4) -> pd.Series:
    """
    Rolling z-score of a time series vs rolling mean/std.
    """
    if series.empty:
        return pd.Series(dtype=float)
    mu = series.rolling(window, min_periods=1).mean()
    sigma = series.rolling(window, min_periods=1).std().replace(0, np.nan)
    z = ((series - mu) / sigma).fillna(0.0)
    return z


def detect_changepoints(series: pd.Series, min_gap: int = 2, z_threshold: float = 1.75) -> list[int]:
    """
    Simple, robust changepoint heuristic:
      - compute burst z-score and year-to-year growth
      - mark years where |z| is large and the growth sign flips vs prior year
    Returns list of years (index values) flagged as changepoints.
    """
    if series.empty or series.size < 4:
        return []
    years = series.index.to_list()
    vals = series.values.astype(np.float64)
    z = burst_scores(series).values.astype(np.float64)
    growth = np.zeros_like(vals)
    for i in range(1, len(vals)):
        growth[i] = (vals[i] - vals[i - 1]) / max(1.0, vals[i - 1])
    cps: list[int] = []
    last_cp_year: int | None = None
    for i in range(2, len(vals)):
        if abs(z[i]) < z_threshold:
            continue
        if growth[i] == 0 or growth[i - 1] == 0:
            continue
        if (growth[i] > 0) == (growth[i - 1] > 0):
            continue
        y = int(years[i])
        if last_cp_year is None or (y - last_cp_year) >= min_gap:
            cps.append(y)
            last_cp_year = y
    return cps


def classify_topic_phase(series: pd.Series) -> str:
    """
    Classify lifecycle phase from recent trend.
    """
    if series.empty:
        return "unknown"
    vals = series.values.astype(np.float64)
    if vals.sum() == 0:
        return "unknown"
    # Recent slope from last 3 points
    tail = vals[-3:] if len(vals) >= 3 else vals
    if len(tail) >= 2:
        slope = float(tail[-1] - tail[0]) / float(len(tail) - 1)
    else:
        slope = 0.0
    recent = float(tail[-1])
    baseline = float(np.mean(vals[:-3])) if len(vals) > 3 else float(np.mean(vals))
    # Heuristics
    if recent <= 1 and baseline <= 1 and slope > 0.25:
        return "emerging"
    if slope > 0.75:
        return "accelerating"
    if slope < -0.75:
        return "fading"
    return "mature"


def topic_lifecycle_metrics(df_papers: pd.DataFrame, centrality_map: dict[str, float] | None = None) -> dict[str, Any]:
    """
    Computes per-topic burst and changepoint analytics from paper(year, cluster_id) counts.
    Optionally uses centrality_map to produce a weighted time series as well.
    """
    subset = df_papers.dropna(subset=["year", "cluster_id"]).copy()
    if subset.empty:
        return {"available": False, "reason": "no (year, cluster_id) rows"}

    subset["year"] = subset["year"].astype(int)
    subset["cluster_id"] = subset["cluster_id"].astype(int)

    min_year = int(subset["year"].min())
    max_year = int(subset["year"].max())
    years = list(range(min_year, max_year + 1))

    # Precompute weighted value per paper
    if centrality_map is None:
        subset["_w"] = 1.0
    else:
        subset["_w"] = subset["paper_id"].map(lambda pid: float(centrality_map.get(pid, 1.0)))

    per_cluster: list[dict[str, Any]] = []
    burst_events: list[dict[str, Any]] = []
    for cid, group in subset.groupby("cluster_id"):
        counts = group.groupby("year").size().reindex(years, fill_value=0).astype(float)
        weighted = group.groupby("year")["_w"].sum().reindex(years, fill_value=0.0).astype(float)
        z = burst_scores(counts)
        zw = burst_scores(weighted)
        cps = detect_changepoints(counts)
        phase = classify_topic_phase(counts)
        burst_years = [int(y) for y in z.index[z.values >= 2.0].tolist()]
        for y, val in z.items():
            if val >= 2.0:
                burst_events.append({"cluster_id": int(cid), "year": int(y), "burst_z": float(val), "count": int(counts.loc[y])})
        per_cluster.append(
            {
                "cluster_id": int(cid),
                "phase": phase,
                "burst_years": burst_years,
                "changepoints": cps,
                "latest_year": int(max_year),
                "latest_count": int(counts.iloc[-1]),
                "latest_burst_z": float(z.iloc[-1]),
                "latest_weighted_burst_z": float(zw.iloc[-1]),
            }
        )
    per_cluster.sort(key=lambda x: (x["phase"] != "emerging", -(x.get("latest_burst_z") or 0.0), -(x.get("latest_count") or 0)))

    return {"available": True, "per_cluster": per_cluster, "burst_events": burst_events[:500]}


# ---------- Bridge / recombination analytics ----------

def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    na = np.linalg.norm(A, axis=1, keepdims=True)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    A = A / na
    B = B / nb
    return A @ B.T


def build_topic_graph(centroids: dict[int, np.ndarray], top_k: int = 5, min_sim: float = 0.15) -> dict[str, Any]:
    """
    Build a sparse cluster graph (undirected) by cosine similarity between cluster centroids.
    Returns {nodes:[cluster_id], edges:[{src,dst,weight}]}
    """
    if not centroids:
        return {"available": False, "reason": "no centroids"}
    node_ids = sorted(centroids.keys())
    X = np.stack([centroids[cid] for cid in node_ids], axis=0).astype(np.float32)
    sims = _cosine_sim_matrix(X, X)
    np.fill_diagonal(sims, -np.inf)
    edges: list[dict[str, Any]] = []
    for i, cid in enumerate(node_ids):
        row = sims[i]
        k_eff = max(0, min(top_k, len(node_ids) - 1))
        if k_eff == 0:
            continue
        idx = np.argpartition(-row, k_eff)[:k_eff]
        ordered = idx[np.argsort(-row[idx])]
        for j in ordered:
            w = float(row[j])
            if w < min_sim or math.isnan(w):
                continue
            other = int(node_ids[int(j)])
            if cid == other:
                continue
            if cid < other:
                edges.append({"src": int(cid), "dst": other, "weight": w})
            else:
                edges.append({"src": other, "dst": int(cid), "weight": w})
    # Deduplicate
    seen: set[tuple[int, int]] = set()
    uniq: list[dict[str, Any]] = []
    for e in sorted(edges, key=lambda x: -x["weight"]):
        key = (int(e["src"]), int(e["dst"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return {"available": True, "nodes": node_ids, "edges": uniq}


def betweenness_centrality_unweighted(nodes: list[int], edges: list[dict[str, Any]]) -> dict[int, float]:
    """
    Brandes betweenness centrality for an unweighted undirected graph.
    """
    if not nodes:
        return {}
    adj: dict[int, list[int]] = {int(n): [] for n in nodes}
    for e in edges:
        a = int(e["src"])
        b = int(e["dst"])
        if a not in adj or b not in adj:
            continue
        adj[a].append(b)
        adj[b].append(a)
    Cb: dict[int, float] = {int(v): 0.0 for v in nodes}
    for s in nodes:
        stack: list[int] = []
        pred: dict[int, list[int]] = {v: [] for v in nodes}
        sigma: dict[int, float] = {v: 0.0 for v in nodes}
        dist: dict[int, int] = {v: -1 for v in nodes}
        sigma[s] = 1.0
        dist[s] = 0
        queue: list[int] = [s]
        q_i = 0
        while q_i < len(queue):
            v = queue[q_i]
            q_i += 1
            stack.append(v)
            for w in adj.get(v, []):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta: dict[int, float] = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta_v = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                else:
                    delta_v = 0.0
                delta[v] += delta_v
            if w != s:
                Cb[w] += delta[w]
    # Normalize for undirected graph
    n = len(nodes)
    if n > 2:
        scale = 1.0 / ((n - 1) * (n - 2) / 2.0)
        for v in Cb:
            Cb[v] *= scale
    return Cb


def clustering_coefficients(nodes: list[int], edges: list[dict[str, Any]]) -> dict[int, float]:
    adj: dict[int, set[int]] = {int(n): set() for n in nodes}
    for e in edges:
        a = int(e["src"])
        b = int(e["dst"])
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    coeff: dict[int, float] = {}
    for v in nodes:
        neigh = list(adj[v])
        d = len(neigh)
        if d < 2:
            coeff[v] = 0.0
            continue
        links = 0
        neigh_set = adj[v]
        for i in range(d):
            a = neigh[i]
            for j in range(i + 1, d):
                b = neigh[j]
                if b in adj.get(a, set()):
                    links += 1
        coeff[v] = float(2 * links / (d * (d - 1)))
    return coeff


def structural_holes_metrics(nodes: list[int], edges: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """
    Burt-style structural holes proxies on an undirected weighted graph.

    Returns dict: node_id -> {"constraint": c, "effective_size": e}

    Notes:
      - Uses row-normalized weights p_ij = w_ij / sum_k w_ik.
      - Constraint: sum_j (p_ij + sum_q p_iq * p_qj)^2
      - Effective size: sum_j (1 - sum_q p_iq * p_qj)
    """
    if not nodes:
        return {}
    adj_w: dict[int, dict[int, float]] = {int(n): {} for n in nodes}
    for e in edges:
        try:
            a = int(e["src"])
            b = int(e["dst"])
            w = float(e.get("weight", 0.0))
        except Exception:
            continue
        if w <= 0 or math.isnan(w):
            continue
        if a in adj_w and b in adj_w:
            adj_w[a][b] = max(adj_w[a].get(b, 0.0), w)
            adj_w[b][a] = max(adj_w[b].get(a, 0.0), w)

    # Row-normalized probabilities p_ij
    P: dict[int, dict[int, float]] = {}
    for i in nodes:
        neigh = adj_w.get(int(i), {})
        s = float(sum(neigh.values()))
        if s <= 0:
            P[int(i)] = {}
        else:
            P[int(i)] = {int(j): float(w / s) for j, w in neigh.items()}

    out: dict[int, dict[str, float]] = {}
    for i in nodes:
        i = int(i)
        pi = P.get(i, {})
        if not pi:
            out[i] = {"constraint": 0.0, "effective_size": 0.0}
            continue
        constraint = 0.0
        eff_size = 0.0
        for j, p_ij in pi.items():
            indirect = 0.0
            for q, p_iq in pi.items():
                indirect += p_iq * P.get(int(q), {}).get(int(j), 0.0)
            constraint += float((p_ij + indirect) ** 2)
            eff_size += float(1.0 - indirect)
        out[i] = {"constraint": float(constraint), "effective_size": float(eff_size)}
    return out


def paper_recombination_metrics(
    df_papers: pd.DataFrame,
    knn: list[dict[str, Any]],
    k: int = 8,
) -> dict[str, Any]:
    """
    Paper recombination index: entropy of neighbor cluster IDs in kNN space.
    """
    cluster_by_paper = {str(row.paper_id): int(row.cluster_id) for _, row in df_papers.iterrows() if row.cluster_id is not None and not pd.isna(row.cluster_id)}
    all_clusters = sorted(set(cluster_by_paper.values()))
    if not all_clusters:
        return {"available": False, "reason": "no cluster labels"}
    denom = float(np.log(max(2, len(all_clusters))))
    per_paper: list[dict[str, Any]] = []
    for item in knn:
        pid = str(item.get("paper_id"))
        neigh = item.get("neighbors") or []
        if pid not in cluster_by_paper:
            continue
        counts: dict[int, int] = {}
        for n in neigh[:k]:
            nid = n.get("paper_id")
            if nid is None:
                continue
            cid = cluster_by_paper.get(str(nid))
            if cid is None:
                continue
            counts[cid] = counts.get(cid, 0) + 1
        total = sum(counts.values())
        if total == 0:
            ent = 0.0
        else:
            p = np.array([v / total for v in counts.values()], dtype=np.float64)
            ent = float(-(p * np.log(p + 1e-12)).sum() / denom) if denom > 0 else 0.0
        top_clusters = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        per_paper.append(
            {
                "paper_id": pid,
                "recombination": float(ent),
                "neighbor_cluster_counts": [{"cluster_id": int(c), "count": int(ct)} for c, ct in top_clusters],
            }
        )
    per_paper.sort(key=lambda x: x["recombination"], reverse=True)
    return {"available": True, "k": int(k), "per_paper": per_paper}


# ---------- Citations / references ----------

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s\)\]\}]+", re.IGNORECASE)
_ARXIV_RE = re.compile(r"\barXiv:\s*\d{4}\.\d{4,5}\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _extract_references_from_text(text: str, max_lines: int = 350) -> list[str]:
    if not text:
        return []
    # Find last "References" section marker
    markers = list(re.finditer(r"(?im)^\s*(references|bibliography|literature cited)\s*$", text))
    if not markers:
        # Try inline markers
        markers = list(re.finditer(r"(?im)\n\s*(references|bibliography|literature cited)\s*\n", text))
    if not markers:
        return []
    start = markers[-1].end()
    tail = text[start:]
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    lines = lines[:max_lines]
    if not lines:
        return []
    entries: list[str] = []
    cur: list[str] = []
    start_re = re.compile(r"^\s*(\[\d+\]|\d+\.)\s+")
    for ln in lines:
        if start_re.match(ln) and cur:
            entries.append(_norm_text(" ".join(cur)))
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        entries.append(_norm_text(" ".join(cur)))
    return entries


def _guess_ref_title(ref: str) -> str | None:
    if not ref:
        return None
    # Prefer quoted titles
    m = re.findall(r"“([^”]{10,220})”", ref)
    if not m:
        m = re.findall(r"\"([^\"]{10,220})\"", ref)
    if m:
        return _norm_text(max(m, key=len))
    # Heuristic: split by periods and pick the longest mid segment
    parts = [p.strip() for p in ref.split(".") if p.strip()]
    if len(parts) >= 2:
        cand = max(parts[1: min(4, len(parts))], key=len, default="")
        if len(cand) >= 12:
            return _norm_text(cand)
    return None


def extract_reference_tables(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """
    Returns (refs_rows, refs_by_paper) where refs_rows entries:
      {paper_id, ref_index, ref_text, ref_title_guess, year_guess, doi_guess, url_guess, arxiv_guess}
    """
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for c in chunks or []:
        pid = c.get("paperID") or c.get("paper_id") or c.get("paperID".lower())
        if pid is None:
            continue
        by_paper.setdefault(str(pid), []).append(c)
    refs_rows: list[dict[str, Any]] = []
    refs_by_paper: dict[str, list[dict[str, Any]]] = {}
    for pid, lst in by_paper.items():
        # Keep order; chunks.json already stores "order"
        lst_sorted = sorted(lst, key=lambda x: int(x.get("order", 0)))
        text = "\n".join(_norm_text(str(x.get("text") or "")) for x in lst_sorted if x.get("text"))
        entries = _extract_references_from_text(text)
        if not entries:
            continue
        paper_refs: list[dict[str, Any]] = []
        for idx, ref in enumerate(entries):
            doi = _DOI_RE.search(ref)
            url = _URL_RE.search(ref)
            arx = _ARXIV_RE.search(ref)
            year = None
            ym = _YEAR_RE.search(ref)
            if ym:
                try:
                    year = int(ym.group(0))
                except Exception:
                    year = None
            title_guess = _guess_ref_title(ref)
            row = {
                "paper_id": pid,
                "ref_index": int(idx),
                "ref_text": ref,
                "ref_title_guess": title_guess,
                "year_guess": year,
                "doi_guess": doi.group(0) if doi else None,
                "url_guess": url.group(0) if url else None,
                "arxiv_guess": arx.group(0) if arx else None,
            }
            refs_rows.append(row)
            paper_refs.append(row)
        refs_by_paper[pid] = paper_refs
    return refs_rows, refs_by_paper


def match_in_corpus_citations(
    refs_rows: list[dict[str, Any]],
    df_papers: pd.DataFrame,
    min_jaccard: float = 0.62,
    min_shared_tokens: int = 3,
) -> list[dict[str, Any]]:
    """
    Link ref rows to in-corpus papers by fuzzy title token overlap.
    Emits edges {src_paper_id, dst_paper_id, match_score, year_guess}.
    """
    if not refs_rows:
        return []
    paper_title_tokens: dict[str, set[str]] = {str(r.paper_id): _token_set(str(r.title)) for _, r in df_papers.iterrows()}
    paper_year: dict[str, int | None] = {str(r.paper_id): (int(r.year) if r.year is not None and not (isinstance(r.year, float) and math.isnan(r.year)) else None) for _, r in df_papers.iterrows()}

    # Inverted index token -> candidate papers
    inv: dict[str, set[str]] = {}
    for pid, toks in paper_title_tokens.items():
        for t in toks:
            inv.setdefault(t, set()).add(pid)

    edges: list[dict[str, Any]] = []
    for ref in refs_rows:
        src = str(ref.get("paper_id"))
        text = ref.get("ref_title_guess") or ref.get("ref_text") or ""
        toks = _token_set(str(text))
        if not toks:
            continue
        candidates: set[str] = set()
        for t in toks:
            candidates.update(inv.get(t, set()))
        best_pid: str | None = None
        best = 0.0
        best_shared = 0
        for pid in candidates:
            ptoks = paper_title_tokens.get(pid, set())
            shared = len(toks.intersection(ptoks))
            if shared < min_shared_tokens:
                continue
            score = _jaccard(toks, ptoks)
            if score > best:
                # Optional year consistency check
                yg = ref.get("year_guess")
                if yg is not None and paper_year.get(pid) is not None:
                    if abs(int(yg) - int(paper_year[pid])) > 1:
                        continue
                best = score
                best_shared = shared
                best_pid = pid
        if best_pid is None or best < min_jaccard:
            continue
        edges.append(
            {
                "src_paper_id": src,
                "dst_paper_id": best_pid,
                "match_score": float(best),
                "year_guess": ref.get("year_guess"),
            }
        )
    # Deduplicate
    uniq: dict[tuple[str, str], dict[str, Any]] = {}
    for e in edges:
        key = (e["src_paper_id"], e["dst_paper_id"])
        prev = uniq.get(key)
        if prev is None or float(e["match_score"]) > float(prev.get("match_score", 0.0)):
            uniq[key] = e
    return list(uniq.values())


def citation_graph_metrics(
    cite_edges: list[dict[str, Any]],
    df_papers: pd.DataFrame,
    max_pairs: int = 120,
) -> dict[str, Any]:
    """
    Compute in-corpus citation graph metrics.
    """
    if not cite_edges:
        return {"available": False, "reason": "no in-corpus cites"}
    ids = [str(pid) for pid in df_papers.paper_id.tolist()]
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    edges_idx: list[tuple[int, int]] = []
    weights: list[float] = []
    for e in cite_edges:
        s = str(e["src_paper_id"])
        d = str(e["dst_paper_id"])
        if s not in id_to_idx or d not in id_to_idx:
            continue
        edges_idx.append((id_to_idx[s], id_to_idx[d]))
        weights.append(float(e.get("match_score", 1.0)))
    pr = pagerank(len(ids), edges_idx, weights=weights)
    in_deg: dict[str, int] = {pid: 0 for pid in ids}
    out_deg: dict[str, int] = {pid: 0 for pid in ids}
    for s, d in edges_idx:
        out_deg[ids[s]] += 1
        in_deg[ids[d]] += 1
    pagerank_rows = [{"paper_id": ids[i], "pagerank": float(pr[i]), "in_degree": int(in_deg[ids[i]]), "out_degree": int(out_deg[ids[i]])} for i in range(len(ids))]
    pagerank_rows.sort(key=lambda x: x["pagerank"], reverse=True)

    # Bibliographic coupling: shared outgoing citations
    out_map: dict[str, set[str]] = {pid: set() for pid in ids}
    in_map: dict[str, set[str]] = {pid: set() for pid in ids}
    for s, d in edges_idx:
        src = ids[s]
        dst = ids[d]
        out_map[src].add(dst)
        in_map[dst].add(src)

    coupling_pairs: list[dict[str, Any]] = []
    cocite_pairs: list[dict[str, Any]] = []
    if len(ids) <= 1500:
        for i in range(len(ids)):
            a = ids[i]
            for j in range(i + 1, len(ids)):
                b = ids[j]
                inter = len(out_map[a].intersection(out_map[b]))
                if inter > 0:
                    coupling_pairs.append({"a": a, "b": b, "shared_refs": int(inter)})
        coupling_pairs.sort(key=lambda x: x["shared_refs"], reverse=True)
        coupling_pairs = coupling_pairs[:max_pairs]

        # Co-citation between cited papers: shared incoming sources
        cited = [pid for pid in ids if in_map[pid]]
        for i in range(len(cited)):
            a = cited[i]
            for j in range(i + 1, len(cited)):
                b = cited[j]
                inter = len(in_map[a].intersection(in_map[b]))
                if inter > 0:
                    cocite_pairs.append({"a": a, "b": b, "shared_citers": int(inter)})
        cocite_pairs.sort(key=lambda x: x["shared_citers"], reverse=True)
        cocite_pairs = cocite_pairs[:max_pairs]

    # Foundational papers per topic
    cluster_by_paper = {str(r.paper_id): (int(r.cluster_id) if r.cluster_id is not None and not pd.isna(r.cluster_id) else None) for _, r in df_papers.iterrows()}
    foundational: dict[int, list[dict[str, Any]]] = {}
    for row in pagerank_rows[: min(500, len(pagerank_rows))]:
        pid = row["paper_id"]
        cid = cluster_by_paper.get(pid)
        if cid is None:
            continue
        foundational.setdefault(int(cid), []).append(row)
    for cid in list(foundational.keys()):
        foundational[cid] = foundational[cid][:5]

    return {
        "available": True,
        "edge_count": len(edges_idx),
        "pagerank": pagerank_rows[:200],
        "top_in_degree": sorted(pagerank_rows, key=lambda x: x["in_degree"], reverse=True)[:200],
        "bibliographic_coupling": coupling_pairs,
        "co_citation": cocite_pairs,
        "foundational_by_cluster": {str(k): v for k, v in foundational.items()},
    }


# ---------- Claim-level controversy / maturity / gaps ----------

def _extract_gap_sentences(text: str, limit: int = 3) -> list[str]:
    if not text:
        return []
    cues = [
        "future work",
        "limitations",
        "limitation",
        "we leave",
        "we do not",
        "not evaluated",
        "not tested",
        "open question",
        "remains",
        "remain",
        "needs",
        "need to",
    ]
    s = _norm_text(text)
    # Naive sentence split
    parts = re.split(r"(?<=[\.\!\?])\s+", s)
    out: list[str] = []
    for p in parts:
        low = p.lower()
        if any(c in low for c in cues):
            out.append(p.strip())
        if len(out) >= limit:
            break
    return out


def claim_edge_controversy_metrics(claim_edge_snapshot: dict[str, Any] | None, df_papers: pd.DataFrame) -> dict[str, Any]:
    """
    Topic-level disagreement using ClaimGraph edge kinds persisted by the Swift app (Output/analytics/claim_edges.json).

    Returns:
      - available
      - generated_at
      - per_cluster: rows compatible with Swift's ClaimsSection.ControversyCluster
      - top_contradictions: representative contradict edges (paper IDs + rationale)
    """
    if not claim_edge_snapshot:
        return {"available": False, "reason": "claim_edges.json not found"}

    edges = claim_edge_snapshot.get("edges")
    if not isinstance(edges, list) or not edges:
        return {"available": False, "reason": "no edges in claim_edges.json"}

    cluster_by_paper: dict[str, int] = {}
    for _, r in df_papers[df_papers.cluster_id.notna()].iterrows():
        try:
            cluster_by_paper[str(r.paper_id)] = int(r.cluster_id)
        except Exception:
            continue

    per_cluster_counts: dict[int, dict[str, Any]] = {}
    top_contradictions: list[dict[str, Any]] = []

    for e in edges[:250000]:
        if not isinstance(e, dict):
            continue
        src = str(e.get("srcPaperID") or e.get("src_paper_id") or "")
        dst = str(e.get("dstPaperID") or e.get("dst_paper_id") or "")
        if not src or not dst:
            continue
        csrc = cluster_by_paper.get(src)
        cdst = cluster_by_paper.get(dst)
        if csrc is None or cdst is None:
            continue

        kind = e.get("kind")
        if isinstance(kind, dict):
            kind = kind.get("rawValue") or kind.get("kind")
        kind_s = str(kind) if kind is not None else ""
        kind_s = kind_s.strip()

        same_cluster = (csrc == cdst)
        supportive = kind_s in {"supports", "extends"}
        contradictory = kind_s == "contradicts"
        comparative = kind_s == "comparesTo"

        def ensure_bucket(cluster_id: int) -> dict[str, Any]:
            return per_cluster_counts.setdefault(
                int(cluster_id),
                {
                    "cluster_id": int(cluster_id),
                    "supports": 0,
                    "contradicts": 0,
                    "compares": 0,
                    "cross_contradicts": 0,
                    "claim_ids": set(),
                    "paper_ids": set(),
                },
            )

        if same_cluster:
            bucket = ensure_bucket(int(csrc))
            if supportive:
                bucket["supports"] += 1
            elif contradictory:
                bucket["contradicts"] += 1
            elif comparative:
                bucket["compares"] += 1
            bucket["paper_ids"].add(src)
            bucket["paper_ids"].add(dst)
            scid = e.get("sourceClaimID") or e.get("source_claim_id")
            tcid = e.get("targetClaimID") or e.get("target_claim_id")
            if scid:
                bucket["claim_ids"].add(str(scid))
            if tcid:
                bucket["claim_ids"].add(str(tcid))

            if contradictory and len(top_contradictions) < 120:
                top_contradictions.append(
                    {
                        "src_paper_id": src,
                        "dst_paper_id": dst,
                        "cluster_id": int(csrc),
                        "kind": kind_s,
                        "rationale": e.get("rationale"),
                    }
                )
        else:
            if contradictory:
                # Attribute cross-cluster contradictions to both sides for "debate hotspots"
                ensure_bucket(int(csrc))["cross_contradicts"] += 1
                ensure_bucket(int(cdst))["cross_contradicts"] += 1

    per_cluster: list[dict[str, Any]] = []
    for cid, bucket in per_cluster_counts.items():
        supports = int(bucket.get("supports", 0))
        contradicts = int(bucket.get("contradicts", 0))
        denom = supports + contradicts + 1e-6
        contradiction_rate = float(contradicts / denom)
        consensus_score = float((supports - contradicts) / denom)
        claim_ids = bucket.get("claim_ids") or set()
        paper_ids = bucket.get("paper_ids") or set()
        claim_diversity = float(len(claim_ids) / max(1, len(paper_ids))) if isinstance(claim_ids, set) and isinstance(paper_ids, set) else None
        per_cluster.append(
            {
                "cluster_id": int(cid),
                "supports": supports,
                "contradicts": contradicts,
                "cross_contradicts": int(bucket.get("cross_contradicts", 0)),
                "contradiction_rate": contradiction_rate,
                "claim_diversity": claim_diversity,
                "consensus_score": consensus_score,
            }
        )
    per_cluster.sort(key=lambda x: (x.get("contradiction_rate") or 0.0), reverse=True)

    return {
        "available": True,
        "generated_at": claim_edge_snapshot.get("generatedAt") or claim_edge_snapshot.get("generated_at"),
        "per_cluster": per_cluster[:500],
        "top_contradictions": top_contradictions[:80],
    }


def stress_test_gap_analytics(user_events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate assumption stress-test events into per-topic "missing tests" suggestions.
    Consumes events appended by Swift (event_type == 'assumption_stress_test').
    """
    events = [e for e in (user_events or []) if isinstance(e, dict) and e.get("event_type") == "assumption_stress_test"]
    if not events:
        return {"available": False, "reason": "no assumption_stress_test events"}

    by_cluster: dict[int, dict[str, Any]] = {}

    def _norm(s: str) -> str:
        return _norm_text(s).strip().rstrip(".")

    for evt in events[-5000:]:
        assumption = evt.get("assumption")
        if not isinstance(assumption, str) or not assumption.strip():
            continue
        assumption_n = _norm(assumption)
        clusters = evt.get("clusters") or {}
        if not isinstance(clusters, dict):
            clusters = {}
        suggested = evt.get("suggested_tests") or []
        if not isinstance(suggested, list):
            suggested = []
        suggested_n = [_norm(str(s)) for s in suggested if str(s).strip()]
        affected_claims = int(evt.get("affected_claims", 0) or 0)

        for k, v in clusters.items():
            try:
                cid = int(k)
            except Exception:
                continue
            try:
                weight = int(v)
            except Exception:
                weight = 0
            if weight <= 0:
                weight = 1

            bucket = by_cluster.setdefault(
                cid,
                {
                    "cluster_id": cid,
                    "event_count": 0,
                    "affected_papers": 0,
                    "affected_claims": 0,
                    "assumptions": {},
                    "suggested_tests": {},
                },
            )
            bucket["event_count"] += 1
            bucket["affected_papers"] += weight
            bucket["affected_claims"] += int(max(0, affected_claims))
            bucket["assumptions"][assumption_n] = bucket["assumptions"].get(assumption_n, 0) + weight
            for t in suggested_n:
                if not t:
                    continue
                bucket["suggested_tests"][t] = bucket["suggested_tests"].get(t, 0) + weight

    per_cluster: list[dict[str, Any]] = []
    for cid, b in by_cluster.items():
        top_assumptions = sorted(b["assumptions"].items(), key=lambda x: x[1], reverse=True)[:8]
        top_tests = sorted(b["suggested_tests"].items(), key=lambda x: x[1], reverse=True)[:10]
        per_cluster.append(
            {
                "cluster_id": int(cid),
                "event_count": int(b.get("event_count", 0)),
                "affected_papers": int(b.get("affected_papers", 0)),
                "affected_claims": int(b.get("affected_claims", 0)),
                "top_assumptions": [{"assumption": a, "count": int(c)} for a, c in top_assumptions],
                "top_suggested_tests": [{"text": t, "count": int(c)} for t, c in top_tests],
            }
        )
    per_cluster.sort(key=lambda x: (x.get("affected_papers") or 0), reverse=True)

    return {
        "available": True,
        "event_count": int(len(events)),
        "per_cluster": per_cluster[:400],
    }


def claim_controversy_and_maturity(df_papers: pd.DataFrame, paper_rows: list[PaperRow]) -> dict[str, Any]:
    # Gather claims with cluster ids
    cluster_by_paper = {str(r.paper_id): (int(r.cluster_id) if r.cluster_id is not None and not pd.isna(r.cluster_id) else None) for _, r in df_papers.iterrows()}
    claims: list[dict[str, Any]] = []
    for row in paper_rows:
        cid = cluster_by_paper.get(row.paper_id)
        if cid is None:
            continue
        for c in row.claims or []:
            stmt = c.get("statement") if isinstance(c, dict) else None
            if not isinstance(stmt, str) or not stmt.strip():
                continue
            claims.append(
                {
                    "paper_id": row.paper_id,
                    "cluster_id": int(cid),
                    "statement": stmt.strip(),
                    "year": c.get("year") or row.year,
                    "strength": float(c.get("strength", 0.5)) if isinstance(c.get("strength"), (int, float)) else 0.5,
                    "sign": int(_claim_sign(stmt)),
                }
            )
    if not claims:
        return {"available": False, "reason": "no claims found"}

    # Cluster claims by similarity via TF-IDF and threshold graph components (per topic to keep small)
    controversy_by_cluster: list[dict[str, Any]] = []
    maturity_by_cluster: list[dict[str, Any]] = []
    top_contested: list[dict[str, Any]] = []
    top_mature: list[dict[str, Any]] = []
    gaps_by_cluster: list[dict[str, Any]] = []

    for cid, group in pd.DataFrame(claims).groupby("cluster_id"):
        texts = group["statement"].tolist()
        paper_ids = group["paper_id"].tolist()
        signs = group["sign"].tolist()
        if len(texts) < 2:
            continue
        tfidf = TfidfVectorizer(max_features=1200, ngram_range=(1, 2))
        mat = normalize(tfidf.fit_transform(texts))
        sims = (mat @ mat.T).toarray()
        np.fill_diagonal(sims, 0.0)
        # Edges for controversy: similarity > thr
        thr = 0.22 if len(texts) < 120 else 0.28
        supports = 0
        contradicts = 0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if sims[i, j] < thr:
                    continue
                if int(signs[i]) == int(signs[j]):
                    supports += 1
                else:
                    contradicts += 1
        denom = supports + contradicts + 1e-6
        contradiction_rate = float(contradicts / denom)
        consensus_score = float((supports - contradicts) / denom)
        claim_diversity = float(len(texts) / max(1, len(set(paper_ids))))
        controversy_by_cluster.append(
            {
                "cluster_id": int(cid),
                "supports": int(supports),
                "contradicts": int(contradicts),
                "contradiction_rate": contradiction_rate,
                "claim_diversity": claim_diversity,
                "consensus_score": consensus_score,
            }
        )

        # Claim maturity: connected components in similarity graph
        visited = [False] * len(texts)
        components: list[list[int]] = []
        for i in range(len(texts)):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                v = stack.pop()
                comp.append(v)
                neigh = np.where(sims[v] >= thr)[0]
                for u in neigh:
                    if not visited[int(u)]:
                        visited[int(u)] = True
                        stack.append(int(u))
            components.append(comp)

        maturity_scores: list[float] = []
        for comp in components:
            comp_papers = [paper_ids[i] for i in comp]
            comp_signs = [signs[i] for i in comp]
            unique_papers = len(set(comp_papers))
            # Majority sign support
            pos = sum(1 for s in comp_signs if s > 0)
            neg = sum(1 for s in comp_signs if s < 0)
            contradictions_present = pos > 0 and neg > 0
            majority = max(pos, neg)
            maturity = (majority / max(1, len(comp))) * (unique_papers / max(1, len(set(paper_ids))))
            if contradictions_present:
                maturity *= 0.5
            maturity_scores.append(float(maturity))

            # Representative statement
            rep = max(comp, key=lambda i: float(np.sum(sims[i])), default=comp[0])
            rep_text = texts[int(rep)]
            if contradictions_present:
                top_contested.append({"cluster_id": int(cid), "statement": rep_text, "maturity": float(maturity), "papers": unique_papers})
            elif maturity >= 0.35 and unique_papers >= 2:
                top_mature.append({"cluster_id": int(cid), "statement": rep_text, "maturity": float(maturity), "papers": unique_papers})

        maturity_by_cluster.append(
            {
                "cluster_id": int(cid),
                "avg_maturity": float(np.mean(maturity_scores)) if maturity_scores else 0.0,
                "claim_cluster_count": int(len(components)),
            }
        )

        # Experiment gaps: limitations cues in summaries/claims (lightweight)
        # Aggregate per paper
        paper_texts = []
        for pid in set(paper_ids):
            p_row = df_papers[df_papers.paper_id == pid]
            if p_row.empty:
                continue
            summ = str(p_row.iloc[0].get("summary") or "")
            gaps = _extract_gap_sentences(summ, limit=3)
            paper_texts.extend(gaps)
        # Normalize and count
        counts: dict[str, int] = {}
        for g in paper_texts:
            norm = _norm_text(g)
            if len(norm) < 30:
                continue
            counts[norm] = counts.get(norm, 0) + 1
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]
        if top:
            gaps_by_cluster.append({"cluster_id": int(cid), "top_gaps": [{"text": t, "count": int(c)} for t, c in top]})

    controversy_by_cluster.sort(key=lambda x: x["contradiction_rate"], reverse=True)
    maturity_by_cluster.sort(key=lambda x: x["avg_maturity"], reverse=True)
    top_contested.sort(key=lambda x: x["maturity"])
    top_mature.sort(key=lambda x: -x["maturity"])

    return {
        "available": True,
        "controversy_by_cluster": controversy_by_cluster[:400],
        "maturity_by_cluster": maturity_by_cluster[:400],
        "top_contested_claims": top_contested[:60],
        "top_mature_claims": top_mature[:60],
        "experiment_gaps_by_cluster": gaps_by_cluster[:300],
    }


# ---------- Methods / datasets / rigor / reproducibility ----------

_METRIC_TOKENS = [
    "bleu",
    "rouge",
    "auroc",
    "auc",
    "f1",
    "accuracy",
    "precision",
    "recall",
    "rmse",
    "mse",
    "mae",
    "sharpe",
    "pnl",
]


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    urls = _URL_RE.findall(text)
    # De-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        u = u.rstrip(").,;]}")
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def artifact_flags(urls: list[str], text: str) -> dict[str, Any]:
    lower = (text or "").lower()
    has_code_phrase = any(p in lower for p in ["code available", "open-source", "open source", "github", "gitlab", "we release code", "released code"])
    has_data_phrase = any(p in lower for p in ["data available", "dataset available", "we release data", "released data", "data can be found"])
    has_code_link = any(("github.com" in u or "gitlab.com" in u or "bitbucket.org" in u) for u in urls)
    has_data_link = any(any(tok in u for tok in ["zenodo", "figshare", "kaggle", "osf.io", "dataverse", "data"]) for u in urls)
    has_model_link = any(any(tok in u for tok in ["huggingface.co", "model", "weights"]) for u in urls)
    openness = 0.0
    openness += 0.5 if (has_code_link or has_code_phrase) else 0.0
    openness += 0.4 if (has_data_link or has_data_phrase) else 0.0
    openness += 0.1 if has_model_link else 0.0
    openness = float(min(1.0, openness))
    return {
        "has_code_link": bool(has_code_link or has_code_phrase),
        "has_data_link": bool(has_data_link or has_data_phrase),
        "has_model_link": bool(has_model_link),
        "openness_score": openness,
    }


def rigor_proxy(text: str, urls: list[str]) -> tuple[float, dict[str, Any]]:
    lower = (text or "").lower()
    cues = {
        "ablation": lower.count("ablation"),
        "baseline": lower.count("baseline"),
        "stat_test": int(any(t in lower for t in ["p-value", "p value", "statistically significant", "t-test", "wilcoxon", "anova"])),
        "ci": int(any(t in lower for t in ["confidence interval", "std.", "standard deviation", "error bar"])),
        "limitations": int("limitation" in lower or "limitations" in lower),
        "figures": len(re.findall(r"\bfig(?:ure)?\s*\d+\b", lower)),
        "tables": len(re.findall(r"\btable\s*\d+\b", lower)),
    }
    art = artifact_flags(urls, text)
    # Simple composite score in [0,1]
    score = 0.0
    score += 0.18 * min(1.0, cues["ablation"] / 2.0)
    score += 0.16 * min(1.0, cues["baseline"] / 3.0)
    score += 0.10 * cues["stat_test"]
    score += 0.10 * cues["ci"]
    score += 0.08 * cues["limitations"]
    score += 0.10 * min(1.0, cues["figures"] / 6.0)
    score += 0.08 * min(1.0, cues["tables"] / 4.0)
    score += 0.20 * art["openness_score"]
    score = float(min(1.0, score))
    return score, {**cues, **art}


def extract_entities(text: str, max_each: int = 6) -> dict[str, list[str]]:
    """
    Lightweight extraction of methods/datasets/metrics from text using regex + context cues.
    """
    if not text:
        return {"methods": [], "datasets": [], "metrics": []}
    s = text
    lower = s.lower()
    methods: dict[str, int] = {}
    datasets: dict[str, int] = {}
    metrics: dict[str, int] = {}

    # Metrics (lexicon)
    for m in _METRIC_TOKENS:
        if m in lower:
            metrics[m.upper() if m in {"auc", "auroc", "bleu"} else m] = metrics.get(m, 0) + lower.count(m)

    # Acronyms and proper nouns as candidates
    acr = re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", s)
    for tok in acr:
        # Context windows to categorize
        pos = s.find(tok)
        window = lower[max(0, pos - 60): pos + 60] if pos >= 0 else lower
        if any(k in window for k in ["dataset", "survey", "catalog", "benchmark", "corpus", "data from", "observations from"]):
            datasets[tok] = datasets.get(tok, 0) + 1
        elif any(k in window for k in ["we propose", "we introduce", "we present", "method", "algorithm", "model", "approach"]):
            methods[tok] = methods.get(tok, 0) + 1

    # Named datasets: "dataset <Name>" / "using <Name> dataset"
    for pat in [
        r"(?i)\bdataset\s+(?:called|named|is)?\s*([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4})",
        r"(?i)\bbenchmark\s+(?:called|named|is)?\s*([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4})",
        r"(?i)\busing\s+the\s+([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4})\s+dataset\b",
        r"(?i)\bfrom\s+the\s+([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4})\s+(?:survey|catalog|dataset|corpus)\b",
    ]:
        for m in re.findall(pat, s):
            name = _norm_text(m)
            if len(name) >= 4:
                datasets[name] = datasets.get(name, 0) + 1

    # Methods: "we propose <Name>" etc (capture short phrase)
    for pat in [
        r"(?i)\bwe\s+(?:propose|introduce|present|develop)\s+(?:a|an|the)?\s*([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4})",
        r"(?i)\busing\s+(?:a|an|the)?\s*([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,3})\s+(?:model|method|algorithm)\b",
    ]:
        for m in re.findall(pat, s):
            name = _norm_text(m)
            if len(name) >= 4:
                methods[name] = methods.get(name, 0) + 1

    def top(d: dict[str, int]) -> list[str]:
        return [k for k, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:max_each]]

    return {
        "methods": top(methods),
        "datasets": top(datasets),
        "metrics": top(metrics),
    }


def methods_datasets_adoption(
    df_papers: pd.DataFrame,
    chunks: list[dict[str, Any]],
    per_paper_limit: int = 2,
) -> dict[str, Any]:
    """
    Extract per-paper entities + adoption curves over time.
    Uses title/summary and optionally a small number of early chunks for context.
    """
    # Build chunks-by-paper (only first few chunks to keep cost down)
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for c in chunks or []:
        pid = c.get("paperID") or c.get("paper_id")
        if pid is None:
            continue
        by_paper.setdefault(str(pid), []).append(c)

    per_paper: list[dict[str, Any]] = []
    all_methods: list[tuple[str, int]] = []
    all_datasets: list[tuple[str, int]] = []
    all_metrics: list[tuple[str, int]] = []

    counts_methods: dict[tuple[int, int], int] = {}
    counts_datasets: dict[tuple[int, int], int] = {}
    counts_metrics: dict[tuple[int, int], int] = {}

    # Stable ids for entities
    method_vocab: dict[str, int] = {}
    dataset_vocab: dict[str, int] = {}
    metric_vocab: dict[str, int] = {}

    def ent_id(vocab: dict[str, int], name: str) -> int:
        if name not in vocab:
            vocab[name] = len(vocab)
        return vocab[name]

    for _, row in df_papers.iterrows():
        pid = str(row.paper_id)
        year = row.year
        year_i = int(year) if year is not None and not (isinstance(year, float) and math.isnan(year)) else None
        base_text = "\n".join(
            [
                str(row.title or ""),
                str(row.summary or ""),
                str(row.method_summary or ""),
                str(row.results_summary or ""),
            ]
        )
        # Add a couple of early chunks (abstract/introduction often)
        chunks_list = sorted(by_paper.get(pid, []), key=lambda x: int(x.get("order", 0)))
        extra = "\n".join(str(c.get("text") or "") for c in chunks_list[:per_paper_limit])
        text = _norm_text(base_text + "\n" + extra)

        urls = extract_urls(text)
        ents = extract_entities(text)
        rigor, rigor_signals = rigor_proxy(text, urls)

        per_paper.append(
            {
                "paper_id": pid,
                "methods": ents["methods"],
                "datasets": ents["datasets"],
                "metrics": ents["metrics"],
                "rigor_proxy": float(rigor),
                "rigor_signals": rigor_signals,
                "artifact_links": urls[:12],
                "has_code_link": bool(rigor_signals.get("has_code_link", False)),
                "has_data_link": bool(rigor_signals.get("has_data_link", False)),
                "openness_score": float(rigor_signals.get("openness_score", 0.0)),
            }
        )

        # Adoption curves (year required)
        if year_i is not None:
            for m in ents["methods"]:
                mid = ent_id(method_vocab, m)
                counts_methods[(mid, year_i)] = counts_methods.get((mid, year_i), 0) + 1
            for d in ents["datasets"]:
                did = ent_id(dataset_vocab, d)
                counts_datasets[(did, year_i)] = counts_datasets.get((did, year_i), 0) + 1
            for m in ents["metrics"]:
                mid = ent_id(metric_vocab, m)
                counts_metrics[(mid, year_i)] = counts_metrics.get((mid, year_i), 0) + 1

    def top_entities(vocab: dict[str, int], counts: dict[tuple[int, int], int], limit: int = 40) -> list[tuple[int, str, int]]:
        totals: dict[int, int] = {}
        for (eid, _), c in counts.items():
            totals[eid] = totals.get(eid, 0) + int(c)
        items = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:limit]
        id_to_name = {v: k for k, v in vocab.items()}
        return [(eid, id_to_name.get(eid, str(eid)), total) for eid, total in items]

    top_m = top_entities(method_vocab, counts_methods, limit=40)
    top_d = top_entities(dataset_vocab, counts_datasets, limit=40)
    top_mt = top_entities(metric_vocab, counts_metrics, limit=40)

    def curve(top_list: list[tuple[int, str, int]], counts: dict[tuple[int, int], int]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for eid, name, _ in top_list:
            rows = [(year, int(c)) for (eid2, year), c in counts.items() if eid2 == eid]
            rows.sort(key=lambda x: x[0])
            out.append({"entity": name, "series": [{"year": int(y), "count": int(c)} for y, c in rows]})
        return out

    return {
        "available": True,
        "per_paper": per_paper,
        "adoption": {
            "methods": curve(top_m, counts_methods),
            "datasets": curve(top_d, counts_datasets),
            "metrics": curve(top_mt, counts_metrics),
        },
    }


# ---------- Workflow analytics: coverage, blind spots, Q&A gaps ----------

def workflow_coverage_and_blindspots(
    df_papers: pd.DataFrame,
    Z: np.ndarray,
    user_events: list[dict[str, Any]],
    cluster_centroids: dict[int, np.ndarray],
    max_blindspots: int = 18,
) -> dict[str, Any]:
    if df_papers.empty:
        return {"available": False, "reason": "no papers"}

    opened_count: dict[str, int] = {}
    notes_count: dict[str, int] = {}
    rec_fb: dict[str, int] = {}
    opened_ts: dict[str, list[str]] = {}
    questions: list[str] = []
    for evt in user_events or []:
        et = evt.get("event_type")
        pid = evt.get("paper_id")
        ts = evt.get("timestamp")
        if et == "opened" and pid:
            opened_count[pid] = opened_count.get(pid, 0) + 1
            if isinstance(ts, str):
                opened_ts.setdefault(pid, []).append(ts)
        if et == "note_saved" and pid:
            notes_count[pid] = notes_count.get(pid, 0) + 1
        if et in {"rec_helpful", "rec_not_helpful"} and pid:
            rec_fb[pid] = rec_fb.get(pid, 0) + (1 if et == "rec_helpful" else -1)
        if et in {"qa_question"}:
            q = evt.get("q")
            if isinstance(q, str) and q.strip():
                questions.append(q.strip())

    # Paper-level read score
    read_score: dict[str, float] = {}
    for pid in df_papers.paper_id.tolist():
        pid_s = str(pid)
        o = float(opened_count.get(pid_s, 0))
        n = float(notes_count.get(pid_s, 0))
        score = 1.0 - math.exp(-(0.55 * o + 0.9 * n))
        read_score[pid_s] = float(min(1.0, score))

    # Cluster coverage
    per_cluster: list[dict[str, Any]] = []
    for cid, group in df_papers[df_papers.cluster_id.notna()].groupby("cluster_id"):
        pids = [str(x) for x in group.paper_id.tolist()]
        if not pids:
            continue
        cov = float(sum(read_score.get(pid, 0.0) for pid in pids) / len(pids))
        per_cluster.append({"cluster_id": int(cid), "coverage": cov, "paper_count": int(len(pids))})
    per_cluster.sort(key=lambda x: x["coverage"])

    # Interest embedding = recency-weighted mean of opened papers (in Z space)
    idx_by_pid = {str(pid): i for i, pid in enumerate(df_papers.paper_id.tolist())}
    weights: list[float] = []
    vecs: list[np.ndarray] = []
    now = dt.datetime.now(dt.timezone.utc)
    for pid, tss in opened_ts.items():
        if pid not in idx_by_pid:
            continue
        # pick most recent timestamp
        best = None
        for ts in tss:
            try:
                dtt = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                continue
            if best is None or dtt > best:
                best = dtt
        if best is None:
            continue
        age_days = max(0.0, (now - best).total_seconds() / 86400.0)
        w = math.exp(-age_days / 60.0)  # 60-day half-ish life
        weights.append(w)
        vecs.append(Z[idx_by_pid[pid]])
    profile = None
    if vecs:
        W = np.array(weights, dtype=np.float32).reshape(-1, 1)
        M = np.stack(vecs, axis=0)
        profile = (M * W).sum(axis=0) / max(1e-6, float(W.sum()))

    blindspots: list[dict[str, Any]] = []
    if profile is not None and cluster_centroids:
        prof = profile.astype(np.float32)
        for cid, cent in cluster_centroids.items():
            # find coverage
            cov = next((x["coverage"] for x in per_cluster if x["cluster_id"] == int(cid)), 0.0)
            sim = cosine_similarity(prof, cent)
            score = float(max(0.0, sim) * (1.0 - cov))
            blindspots.append({"cluster_id": int(cid), "interest_similarity": float(sim), "coverage": float(cov), "blindspot_score": score})
        blindspots.sort(key=lambda x: x["blindspot_score"], reverse=True)

    return {
        "available": True,
        "paper_read_scores": [{"paper_id": pid, "read_score": float(sc)} for pid, sc in read_score.items() if sc > 0],
        "cluster_coverage": per_cluster[:600],
        "blindspots": blindspots[:max_blindspots],
        "questions_asked": len(questions),
    }


def qa_gap_analytics(
    user_events: list[dict[str, Any]],
    df_papers: pd.DataFrame,
    max_questions: int = 80,
) -> dict[str, Any]:
    """
    Per-question evidence density / breadth from user_events.
    Uses retrieval metrics if app logged them; otherwise falls back to TF-IDF vs paper summaries.
    """
    if not user_events:
        return {"available": False, "reason": "no user_events"}

    questions: dict[str, dict[str, Any]] = {}
    for evt in user_events:
        et = evt.get("event_type")
        q = evt.get("q")
        if not isinstance(q, str) or not q.strip():
            continue
        qq = q.strip()
        st = questions.setdefault(qq, {"asked": 0, "answered": 0, "retrieval": []})
        if et == "qa_question":
            st["asked"] += 1
        if et == "qa_answer_ready":
            st["answered"] += 1
        if et in {"qa_retrieval"}:
            st["retrieval"].append(evt)

    if not questions:
        return {"available": False, "reason": "no qa_question events"}

    # Build TF-IDF over paper titles+summaries for fallback
    docs = [(str(r.title or "") + "\n" + str(r.summary or "")) for _, r in df_papers.iterrows()]
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    mat = normalize(tfidf.fit_transform(docs)) if docs else None
    cluster_by_pid = {str(r.paper_id): (int(r.cluster_id) if r.cluster_id is not None and not pd.isna(r.cluster_id) else None) for _, r in df_papers.iterrows()}

    per_q: list[dict[str, Any]] = []
    for q, st in questions.items():
        retrieval = st.get("retrieval") or []
        if retrieval:
            # Use most recent logged retrieval payload
            last = retrieval[-1]
            top = last.get("top_scores") or []
            top = [float(x) for x in top if isinstance(x, (int, float))][:6]
            margin = float(top[0] - top[1]) if len(top) >= 2 else float(top[0]) if top else 0.0
            breadth = float(last.get("support_breadth", 0.0)) if isinstance(last.get("support_breadth"), (int, float)) else 0.0
            per_q.append(
                {
                    "question": q,
                    "asked": int(st["asked"]),
                    "answered": int(st["answered"]),
                    "top_score": float(top[0]) if top else 0.0,
                    "margin": margin,
                    "support_breadth": breadth,
                    "unanswered": int(st["answered"]) == 0,
                }
            )
        elif mat is not None:
            qv = normalize(tfidf.transform([q]))
            sims = (mat @ qv.T).toarray().reshape(-1)
            order = np.argsort(-sims)[:6]
            top_scores = sims[order]
            margin = float(top_scores[0] - top_scores[1]) if len(top_scores) >= 2 else float(top_scores[0]) if len(top_scores) >= 1 else 0.0
            # Breadth: entropy over clusters among top papers
            cids = [cluster_by_pid.get(str(df_papers.iloc[int(i)].paper_id)) for i in order]
            counts: dict[int, int] = {}
            for cid in cids:
                if cid is None:
                    continue
                counts[int(cid)] = counts.get(int(cid), 0) + 1
            total = sum(counts.values())
            if total <= 1:
                breadth = 0.0
            else:
                p = np.array([v / total for v in counts.values()], dtype=np.float64)
                breadth = float(-(p * np.log(p + 1e-12)).sum() / np.log(max(2, len(counts))))
            per_q.append(
                {
                    "question": q,
                    "asked": int(st["asked"]),
                    "answered": int(st["answered"]),
                    "top_score": float(top_scores[0]) if len(top_scores) else 0.0,
                    "margin": margin,
                    "support_breadth": float(breadth),
                    "unanswered": int(st["answered"]) == 0,
                }
            )

    per_q.sort(key=lambda x: (not x["unanswered"], -x["asked"], x["top_score"]))
    return {"available": True, "questions": per_q[:max_questions]}


def marginal_information_gain_recommendations(
    df_papers: pd.DataFrame,
    Z: np.ndarray,
    cluster_centroids: dict[int, np.ndarray],
    workflow: dict[str, Any] | None,
    n_select: int = 8,
    candidate_cap: int = 1200,
) -> dict[str, Any]:
    """
    Greedy submodular coverage (facility-location style) over cluster centroids.
    Produces a recommendation list that maximizes *marginal information gain* relative to already-read papers.
    """
    if df_papers.empty or Z.size == 0 or not cluster_centroids:
        return {"available": False, "reason": "missing inputs"}

    idx_by_pid = {str(pid): i for i, pid in enumerate(df_papers.paper_id.tolist())}
    # Read scores from workflow section
    read_score: dict[str, float] = {}
    cluster_coverage: dict[int, float] = {}
    if workflow and workflow.get("available"):
        for row in workflow.get("paper_read_scores", []):
            pid = row.get("paper_id")
            if pid is None:
                continue
            read_score[str(pid)] = float(row.get("read_score", 0.0))
        for row in workflow.get("cluster_coverage", []):
            cid = row.get("cluster_id")
            if cid is None:
                continue
            cluster_coverage[int(cid)] = float(row.get("coverage", 0.0))

    # Normalize paper vectors for cosine
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Z_n = Z / norms

    # Build centroid matrix
    cids = sorted(cluster_centroids.keys())
    C = np.stack([cluster_centroids[cid] for cid in cids], axis=0).astype(np.float32)
    cn = np.linalg.norm(C, axis=1, keepdims=True)
    cn[cn == 0] = 1.0
    Cn = C / cn

    # Already covered papers = read_score >= 0.5 (opened/notes)
    covered_ids = [pid for pid, sc in read_score.items() if sc >= 0.5 and pid in idx_by_pid]
    covered_vecs = Z_n[[idx_by_pid[pid] for pid in covered_ids]] if covered_ids else None
    current_best = np.zeros(len(cids), dtype=np.float32)
    if covered_vecs is not None and covered_vecs.size > 0:
        sims = covered_vecs @ Cn.T  # (m,k)
        current_best = np.max(sims, axis=0).astype(np.float32)

    # Candidate set: unread-ish papers (default all if no events)
    candidates: list[str] = []
    for pid in df_papers.paper_id.tolist():
        pid_s = str(pid)
        sc = read_score.get(pid_s, 0.0)
        if sc < 0.15:
            candidates.append(pid_s)
    if not candidates:
        candidates = [str(pid) for pid in df_papers.paper_id.tolist()]

    # Cap candidates by relevance to uncovered clusters (cheap heuristic)
    if len(candidates) > candidate_cap:
        # Score by similarity to low-coverage centroids
        weights = np.array([1.0 - cluster_coverage.get(cid, 0.0) for cid in cids], dtype=np.float32)
        weights = weights / max(1e-6, float(weights.sum()))
        scored: list[tuple[str, float]] = []
        for pid in candidates:
            v = Z_n[idx_by_pid[pid]]
            sim = float((v @ Cn.T).dot(weights))
            scored.append((pid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        candidates = [pid for pid, _ in scored[:candidate_cap]]

    cluster_weights = np.array([1.0 - cluster_coverage.get(cid, 0.0) for cid in cids], dtype=np.float32)
    if cluster_weights.sum() <= 0:
        cluster_weights = np.ones_like(cluster_weights)

    selected: list[dict[str, Any]] = []
    chosen: set[str] = set()
    for _ in range(max(0, n_select)):
        best_pid = None
        best_gain = 0.0
        best_sim = None
        for pid in candidates:
            if pid in chosen:
                continue
            v = Z_n[idx_by_pid[pid]]
            sim = v @ Cn.T  # (k,)
            gain = float(np.sum(cluster_weights * np.maximum(0.0, sim - current_best)))
            if gain > best_gain:
                best_gain = gain
                best_pid = pid
                best_sim = sim
        if best_pid is None:
            break
        chosen.add(best_pid)
        selected.append({"paper_id": best_pid, "marginal_gain": float(best_gain)})
        if best_sim is not None:
            current_best = np.maximum(current_best, best_sim.astype(np.float32))

    return {"available": True, "selected": selected}


# ---------- Hygiene: duplicates + ingestion diagnostics ----------

def title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_norm = _norm_text(a.lower())
    b_norm = _norm_text(b.lower())
    # Token Jaccard blended with SequenceMatcher ratio
    jac = _jaccard(_token_set(a_norm), _token_set(b_norm))
    seq = difflib.SequenceMatcher(a=a_norm, b=b_norm).ratio()
    return float(0.65 * jac + 0.35 * seq)


def detect_duplicates(
    df_papers: pd.DataFrame,
    Z: np.ndarray,
    knn: list[dict[str, Any]],
    sim_threshold: float = 0.965,
    title_threshold: float = 0.82,
) -> dict[str, Any]:
    if df_papers.empty:
        return {"available": False, "reason": "no papers"}
    title_map = {str(r.paper_id): str(r.title or "") for _, r in df_papers.iterrows()}
    page_map = {str(r.paper_id): (int(r.page_count) if r.page_count is not None and not (isinstance(r.page_count, float) and math.isnan(r.page_count)) else None) for _, r in df_papers.iterrows()}

    pairs: list[dict[str, Any]] = []
    for item in knn:
        pid = str(item.get("paper_id"))
        for neigh in item.get("neighbors", []):
            nid = str(neigh.get("paper_id"))
            if pid >= nid:
                continue
            sim = float(neigh.get("score", 0.0))
            if sim < sim_threshold:
                continue
            ts = title_similarity(title_map.get(pid, ""), title_map.get(nid, ""))
            if ts < title_threshold:
                continue
            pc_a = page_map.get(pid)
            pc_b = page_map.get(nid)
            page_delta = abs(pc_a - pc_b) if pc_a is not None and pc_b is not None else None
            pairs.append(
                {
                    "a": pid,
                    "b": nid,
                    "embedding_sim": sim,
                    "title_sim": ts,
                    "page_count_delta": page_delta,
                }
            )
    pairs.sort(key=lambda x: (-x["embedding_sim"], -x["title_sim"]))

    # Union-Find groups
    parent: dict[str, str] = {}
    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for p in pairs:
        union(p["a"], p["b"])
    groups: dict[str, list[str]] = {}
    for pid in title_map.keys():
        r = find(pid)
        groups.setdefault(r, []).append(pid)
    dup_groups = [g for g in groups.values() if len(g) >= 2]
    dup_groups.sort(key=lambda g: -len(g))

    return {"available": True, "pairs": pairs[:120], "groups": dup_groups[:60]}


def ingestion_quality_diagnostics(df_papers: pd.DataFrame, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for c in chunks or []:
        pid = c.get("paperID") or c.get("paper_id")
        if pid is None:
            continue
        by_paper.setdefault(str(pid), []).append(c)

    issues: list[dict[str, Any]] = []
    missing_year = 0
    missing_title = 0
    low_text = 0
    missing_chunks = 0
    for _, row in df_papers.iterrows():
        pid = str(row.paper_id)
        title = str(row.title or "").strip()
        year = row.year
        summary = str(row.summary or "")
        page_count = row.page_count if row.page_count is not None and not (isinstance(row.page_count, float) and math.isnan(row.page_count)) else None
        chunks_list = by_paper.get(pid, [])
        text_len = int(sum(len(str(c.get("text") or "")) for c in chunks_list))
        flags: list[str] = []
        if not title or len(title) < 5:
            missing_title += 1
            flags.append("missing_title")
        if year is None or (isinstance(year, float) and math.isnan(year)):
            missing_year += 1
            flags.append("missing_year")
        if not chunks_list:
            missing_chunks += 1
            flags.append("missing_chunks")
        if len(summary.strip()) < 60:
            flags.append("short_summary")
        # Text yield heuristic
        if page_count is not None and page_count > 0:
            if text_len < 300 * int(page_count):
                low_text += 1
                flags.append("low_text_yield")
        elif text_len < 1200:
            low_text += 1
            flags.append("low_text_yield")
        if flags:
            issues.append({"paper_id": pid, "flags": flags, "chunk_text_len": text_len, "page_count": page_count})

    issues.sort(key=lambda x: (len(x["flags"]), x.get("chunk_text_len", 0)), reverse=True)
    return {
        "available": True,
        "missing_year": int(missing_year),
        "missing_title": int(missing_title),
        "missing_chunks": int(missing_chunks),
        "low_text_yield": int(low_text),
        "issues": issues[:200],
    }


def _claim_sign(text: str) -> int:
    lower = text.lower()
    negatives = ["not ", "does not", "fail", "fails", "failed", "worse", "without improvement", "no improvement"]
    for tok in negatives:
        if tok in lower:
            return -1
    return 1


def claim_similarity_edges(claims: list[dict[str, Any]], threshold: float = 0.22, time_decay: float = 0.15) -> list[dict[str, Any]]:
    """Build directed edges from older -> newer similar claims using TF-IDF cosine with sign and temporal decay."""
    if not claims:
        return []
    texts = [c.get("statement", "") or "" for c in claims]
    years = [c.get("year") for c in claims]
    paper_ids = [c.get("paper_id") for c in claims]
    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    mat = tfidf.fit_transform(texts)
    mat = normalize(mat)
    sims = mat @ mat.T
    edges: list[dict[str, Any]] = []
    for i in range(sims.shape[0]):
        for j in range(sims.shape[1]):
            if i == j:
                continue
            sim = sims[i, j]
            if sim < threshold or math.isnan(sim):
                continue
            yi = years[i]
            yj = years[j]
            if yi is None or yj is None or yi <= yj:
                continue  # only older -> newer
            dt_years = max(0.0, float(yi - yj))
            decay = math.exp(-time_decay * dt_years)
            sign = _claim_sign(texts[i])
            edges.append(
                {
                    "src": paper_ids[j],
                    "dst": paper_ids[i],
                    "weight": float(sim * decay * abs(sign)),
                    "sign": int(sign),
                    "from_year": yj,
                    "to_year": yi,
                }
            )
    return edges


def compute_influence(df_papers: pd.DataFrame, claims: list[dict[str, Any]], edges_raw: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, float], dict[str, float]]:
    if not claims:
        return [], [], [], {}, {}
    ids = df_papers.paper_id.tolist()
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    # Build edges using claim similarity + assumptions overlap
    if edges_raw is None:
        edges_raw = claim_similarity_edges(claims, threshold=0.18)
    edges_idx: list[tuple[int, int]] = []
    weights_abs: list[float] = []
    weights_pos: list[float] = []
    weights_neg: list[float] = []
    in_strength: dict[str, float] = {pid: 0.0 for pid in ids}
    out_strength: dict[str, float] = {pid: 0.0 for pid in ids}
    for e in edges_raw:
        s = e.get("src")
        d = e.get("dst")
        if s not in id_to_idx or d not in id_to_idx:
            continue
        w = float(abs(e.get("weight", 0.0)))
        sign = int(e.get("sign", 1))
        edges_idx.append((id_to_idx[s], id_to_idx[d]))
        weights_abs.append(w)
        weights_pos.append(w if sign > 0 else 0.0)
        weights_neg.append(w if sign < 0 else 0.0)
        out_strength[s] = out_strength.get(s, 0.0) + w
        in_strength[d] = in_strength.get(d, 0.0) + w

    ranks_abs = pagerank(len(ids), edges_idx, weights_abs)
    ranks_pos = pagerank(len(ids), edges_idx, weights_pos)
    ranks_neg = pagerank(len(ids), edges_idx, weights_neg)

    influence_abs = [{"paper_id": pid, "influence": float(ranks_abs[i])} for i, pid in enumerate(ids)]
    influence_pos = [{"paper_id": pid, "influence": float(ranks_pos[i])} for i, pid in enumerate(ids)]
    influence_neg = [{"paper_id": pid, "influence": float(ranks_neg[i])} for i, pid in enumerate(ids)]
    return influence_abs, influence_pos, influence_neg, out_strength, in_strength


def idea_flow_edges(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # reuse similarity edges; include weights/years
    return claim_similarity_edges(claims, threshold=0.18)


# ---------- Recommendation (simple LinUCB-ish heuristic) ----------

def recommend_papers(loadings: list[dict[str, Any]], novelty: list[dict[str, Any]], centrality: list[dict[str, Any]], k: int = 5) -> list[str]:
    novelty_map = {n["paper_id"]: n["novelty"] for n in novelty}
    cent_map = {c["paper_id"]: c.get("weighted_degree", 0.0) for c in centrality}
    scored = []
    for row in loadings:
        pid = row["paper_id"]
        base = sum(abs(x) for x in row["scores"][:3])  # exploration via factor magnitude
        bonus_n = novelty_map.get(pid, 0.0)
        bonus_c = cent_map.get(pid, 0.0)
        score = 0.6 * base + 0.3 * bonus_n + 0.1 * bonus_c
        scored.append((pid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored[:k]]


# ---------- Uncertainty (bootstrap overlap) ----------

def estimate_answer_confidence(top_sets: list[list[str]]) -> float:
    if not top_sets:
        return 0.0
    base = set(top_sets[0])
    if not base:
        return 0.0
    overlaps = []
    for ts in top_sets[1:]:
        overlaps.append(len(base.intersection(ts)) / max(1, len(base.union(ts))))
    if not overlaps:
        return 1.0
    return float(sum(overlaps) / len(overlaps))


def build_paper_metrics(
    df_papers: pd.DataFrame,
    multi_novelty: dict[str, dict[str, float]],
    comb_novelty: dict[str, float],
    novelty_unc: dict[str, float],
    centrality_deg: dict[str, float],
    eigen_cent: dict[str, float],
    edges: list[dict[str, Any]],
    influence_abs: list[dict[str, Any]],
    influence_pos: list[dict[str, Any]],
    influence_neg: list[dict[str, Any]],
    out_strength: dict[str, float],
    in_strength: dict[str, float],
    drift_contrib_map: dict[str, float],
) -> list[dict[str, Any]]:
    years = {row.paper_id: row.year for _, row in df_papers.iterrows()}
    # consensus structural
    z_deg = zscore(centrality_deg)
    z_eig = zscore(eigen_cent)
    consensus_struct = {pid: 0.6 * z_deg.get(pid, 0.0) + 0.4 * z_eig.get(pid, 0.0) for pid in years.keys()}

    # claim + temporal consensus from edges
    incoming_pos: dict[str, float] = {}
    incoming_neg: dict[str, float] = {}
    outgoing_pos: dict[str, float] = {}
    outgoing_neg: dict[str, float] = {}
    for e in edges:
        src = e.get("src")
        dst = e.get("dst")
        w = float(e.get("weight", 0.0))
        sign = int(e.get("sign", 1))
        if src:
            if sign > 0:
                outgoing_pos[src] = outgoing_pos.get(src, 0.0) + w
            else:
                outgoing_neg[src] = outgoing_neg.get(src, 0.0) + w
        if dst:
            if sign > 0:
                incoming_pos[dst] = incoming_pos.get(dst, 0.0) + w
            else:
                incoming_neg[dst] = incoming_neg.get(dst, 0.0) + w

    def consensus_score(pos_map, neg_map, pid):
        pos = pos_map.get(pid, 0.0)
        neg = neg_map.get(pid, 0.0)
        denom = pos + neg + 1e-6
        return (pos - neg) / denom

    consensus_claim = {pid: consensus_score(incoming_pos, incoming_neg, pid) for pid in years.keys()}
    consensus_temporal = {pid: consensus_score(outgoing_pos, outgoing_neg, pid) for pid in years.keys()}

    consensus_total_raw = {pid: 0.6 * consensus_struct.get(pid, 0.0) + 0.2 * consensus_claim.get(pid, 0.0) + 0.2 * consensus_temporal.get(pid, 0.0) for pid in years.keys()}
    consensus_z = zscore_by_year(consensus_total_raw, years)

    # novelty z-score by year (using cluster distance)
    nov_cluster_map = {pid: vals.get("nov_cluster", 0.0) for pid, vals in multi_novelty.items()}
    nov_z = zscore_by_year(nov_cluster_map, years)

    # Influence maps
    infl_abs_map = {item["paper_id"]: item["influence"] for item in influence_abs}
    infl_pos_map = {item["paper_id"]: item["influence"] for item in influence_pos}
    infl_neg_map = {item["paper_id"]: item["influence"] for item in influence_neg}

    # Roles
    roles = {}
    for pid in years.keys():
        out_w = out_strength.get(pid, 0.0)
        in_w = in_strength.get(pid, 0.0)
        denom = out_w + in_w + 1e-6
        source = out_w / denom
        sink = in_w / denom
        bridge = min(out_w, in_w) / denom
        roles[pid] = (source, bridge, sink)

    metrics: list[dict[str, Any]] = []
    for pid in years.keys():
        mv = multi_novelty.get(pid, {})
        metrics.append(
            {
                "paper_id": pid,
                "nov_cluster": mv.get("nov_cluster", 0.0),
                "nov_global": mv.get("nov_global", 0.0),
                "nov_directional": mv.get("nov_directional", 0.0),
                "nov_combinatorial": comb_novelty.get(pid, 0.0),
                "novelty_uncertainty": novelty_unc.get(pid, 0.0),
                "z_novelty": nov_z.get(pid, 0.0),
                "consensus_struct": consensus_struct.get(pid, 0.0),
                "consensus_claim": consensus_claim.get(pid, 0.0),
                "consensus_temporal": consensus_temporal.get(pid, 0.0),
                "consensus_total": consensus_total_raw.get(pid, 0.0),
                "z_consensus": consensus_z.get(pid, 0.0),
                "consensus_uncertainty": 0.0,
                "influence_abs": infl_abs_map.get(pid, 0.0),
                "influence_pos": infl_pos_map.get(pid, 0.0),
                "influence_neg": infl_neg_map.get(pid, 0.0),
                "drift_contrib": drift_contrib_map.get(pid, 0.0),
                "role_source": roles[pid][0],
                "role_bridge": roles[pid][1],
                "role_sink": roles[pid][2],
            }
        )
    return metrics


# ---------- DuckDB persistence ----------

def persist_duckdb(
    db_path: pathlib.Path,
    df_papers: pd.DataFrame,
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    extra_tables: dict[str, pd.DataFrame] | None = None,
):
    con = duckdb.connect(str(db_path))

    # Papers (metadata only)
    con.execute("CREATE OR REPLACE TABLE papers AS SELECT * FROM df_papers").execute() if False else None
    con.register("df_papers", df_papers)
    con.execute("CREATE OR REPLACE TABLE papers AS SELECT * FROM df_papers")

    # Embeddings
    emb_df = pd.DataFrame(
        {
            "paper_id": df_papers.paper_id,
            "embedding": embeddings.tolist(),
        }
    )
    con.register("df_embeddings", emb_df)
    con.execute("CREATE OR REPLACE TABLE paper_embeddings AS SELECT * FROM df_embeddings")

    # Chunks
    chunk_df = pd.DataFrame(chunks) if chunks else pd.DataFrame(columns=["id", "paperID", "text", "embedding", "order", "pageHint"])
    con.register("df_chunks", chunk_df)
    con.execute("CREATE OR REPLACE TABLE paper_chunks AS SELECT * FROM df_chunks")

    # Claims
    claims_rows: list[dict[str, Any]] = []
    for _, row in df_papers.iterrows():
        for claim in row.claims or []:
            claims_rows.append(
                {
                    "claim_id": claim.get("id"),
                    "paper_id": row.paper_id,
                    "statement": claim.get("statement"),
                    "assumptions": claim.get("assumptions"),
                    "year": claim.get("year") or row.year,
                    "strength": claim.get("strength"),
                }
            )
    claim_df = pd.DataFrame(claims_rows)
    con.register("df_claims", claim_df)
    con.execute("CREATE OR REPLACE TABLE claims AS SELECT * FROM df_claims")

    # Methods
    method_rows: list[dict[str, Any]] = []
    for _, row in df_papers.iterrows():
        if not row.method_pipeline:
            continue
        method_rows.append(
            {
                "paper_id": row.paper_id,
                "pipeline_json": json.dumps(row.method_pipeline),
            }
        )
    method_df = pd.DataFrame(method_rows)
    con.register("df_methods", method_df)
    con.execute("CREATE OR REPLACE TABLE methods AS SELECT * FROM df_methods")

    # Export quick Parquet snapshots for notebooks / Swift reloads.
    out_dir = db_path.parent / "analytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY papers TO '{out_dir / 'papers.parquet'}' (FORMAT PARQUET, CODEC 'ZSTD')")
    con.execute(f"COPY paper_embeddings TO '{out_dir / 'paper_embeddings.parquet'}' (FORMAT PARQUET, CODEC 'ZSTD')")
    con.execute(f"COPY paper_chunks TO '{out_dir / 'paper_chunks.parquet'}' (FORMAT PARQUET, CODEC 'ZSTD')")
    con.execute(f"COPY claims TO '{out_dir / 'claims.parquet'}' (FORMAT PARQUET, CODEC 'ZSTD')")
    con.execute(f"COPY methods TO '{out_dir / 'methods.parquet'}' (FORMAT PARQUET, CODEC 'ZSTD')")

    # Optional extra tables (refs, in_corpus_cites, etc.)
    if extra_tables:
        for name, df in extra_tables.items():
            if df is None:
                continue
            if getattr(df, "shape", (0, 0))[1] == 0:
                continue
            view = f"df_{name}"
            con.register(view, df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {view}")
            con.execute(f"COPY {name} TO '{out_dir / f'{name}.parquet'}' (FORMAT PARQUET, CODEC 'ZSTD')")
    con.close()

# ---------- Summary export ----------

def write_summary(analytics_dir: pathlib.Path, summary: dict[str, Any]):
    out_path = analytics_dir / "analytics.json"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[analytics] Wrote summary to {out_path}")

# ---------- Main entry ----------

def main():
    parser = argparse.ArgumentParser(description="Rebuild analytics DuckDB + JSON outputs.")
    parser.add_argument("--base", help="Repo root (defaults to script parent)", default=None)
    parser.add_argument("--root", dest="base", help="Alias for --base", default=None)
    parser.add_argument("--db", help="DuckDB file path (default Output/atlas.duckdb)", default=None)
    parser.add_argument("--counterfactual-cutoffs", nargs="*", type=int, default=[2010, 2015, 2020], help="Year cutoffs for counterfactual scenarios.")
    args = parser.parse_args()

    paths = resolve_paths(args.base)
    db_path = pathlib.Path(args.db).expanduser() if args.db else paths["db_path"]

    papers_dir = paths["papers_dir"]
    if not papers_dir.exists():
        raise SystemExit(f"No papers found at {papers_dir}. Run the Swift app ingestion first.")

    print(f"[analytics] Loading papers from {papers_dir}")
    paper_rows, embedding_lists = load_papers(papers_dir)
    if not paper_rows:
        raise SystemExit("No papers with embeddings to load.")

    embeddings = np.asarray(embedding_lists, dtype=np.float32)
    df_papers = pd.DataFrame([row.__dict__ for row in paper_rows])
    tag_texts = [tag_text_for_row(row) for row in paper_rows]
    corpus_version = compute_corpus_version(paper_rows)
    galaxy_layout = load_galaxy_layout(paths["output"], corpus_version)

    # Whiten embeddings for isotropic distance metrics
    Z_whitened, pca_model = whiten_embeddings(embeddings)

    print(f"[analytics] Loaded {len(df_papers)} papers; building DuckDB at {db_path}")
    chunks = load_chunks(paths["chunks_path"])
    user_events = load_user_events(paths["output"])
    claim_edge_snapshot = load_claim_edge_snapshot(paths["output"])

    topic_trends = compute_topic_trends(df_papers)

    drift = compute_drift(df_papers, Z_whitened)
    drift_vol = drift_volatility(drift)
    drift_vec_map = drift_vector_map(df_papers, Z_whitened)

    cluster_centroids_z = compute_cluster_centroids(df_papers, Z_whitened)

    multi_nov = compute_multi_novelty(df_papers, Z_whitened, drift_vec_map)
    nov_unc = novelty_uncertainty(df_papers, Z_whitened, cluster_centroids_z)
    comb_nov = combinational_novelty(df_papers)

    knn = build_knn(Z_whitened, df_papers.paper_id.tolist(), k=8)
    centrality = centrality_from_knn(knn)
    eigen_cent = eigen_centrality_from_knn(knn)
    centrality_map = {c["paper_id"]: float(c.get("weighted_degree", 0.0)) for c in centrality}

    # Factor model
    factors, loadings, factor_labels = compute_factor_loadings(embeddings, df_papers.paper_id.tolist(), tag_texts, n_factors=8)
    factor_exposures = factor_exposures_over_time(loadings, df_papers.year.tolist(), n_factors=8)
    factor_exposures_user = factor_exposures_from_reads(loadings, df_papers, user_events, n_factors=8)

    # Influence via claims
    all_claims: list[dict[str, Any]] = []
    for row in paper_rows:
        if row.claims:
            for c in row.claims:
                all_claims.append({
                    "paper_id": row.paper_id,
                    "statement": c.get("statement"),
                    "assumptions": c.get("assumptions", []),
                    "year": c.get("year") or row.year,
                })
    claim_edges = claim_similarity_edges(all_claims, threshold=0.18)
    influence_abs, influence_pos, influence_neg, out_strength, in_strength = compute_influence(df_papers, all_claims, edges_raw=claim_edges)

    # Higher-level analytics additions
    stability = cluster_stability_multi_seed_kmeans(df_papers, Z_whitened, runs=20, seed=0)
    stability_boundary = cluster_stability_and_boundary(df_papers, Z_whitened, trials=10, noise=0.01)
    map_quality = map_quality_metrics(galaxy_layout, df_papers, Z_whitened, k=15)
    paper_map_quality, paper_map_distortion = paper_layout_quality_metrics(df_papers, Z_whitened, k=15, max_n=4000, seed=0)
    lifecycle = topic_lifecycle_metrics(df_papers, centrality_map=centrality_map)

    topic_graph = build_topic_graph(cluster_centroids_z, top_k=5, min_sim=0.15)
    topic_metrics: list[dict[str, Any]] = []
    if topic_graph.get("available"):
        nodes = topic_graph.get("nodes") or []
        edges = topic_graph.get("edges") or []
        btw = betweenness_centrality_unweighted(nodes, edges)
        cc = clustering_coefficients(nodes, edges)
        holes = structural_holes_metrics(nodes, edges)
        deg: dict[int, int] = {int(n): 0 for n in nodes}
        for e in edges:
            deg[int(e["src"])] = deg.get(int(e["src"]), 0) + 1
            deg[int(e["dst"])] = deg.get(int(e["dst"]), 0) + 1
        for cid in nodes:
            b = float(btw.get(int(cid), 0.0))
            cl = float(cc.get(int(cid), 0.0))
            constraint = float(holes.get(int(cid), {}).get("constraint", 0.0))
            eff_size = float(holes.get(int(cid), {}).get("effective_size", 0.0))
            topic_metrics.append(
                {
                    "cluster_id": int(cid),
                    "betweenness": b,
                    "clustering_coeff": cl,
                    "bridging_centrality": float(b * (1.0 - cl)),
                    "degree": int(deg.get(int(cid), 0)),
                    "constraint": constraint,
                    "effective_size": eff_size,
                }
            )
        topic_metrics.sort(key=lambda x: x["bridging_centrality"], reverse=True)

    recombination = paper_recombination_metrics(df_papers, knn, k=8)

    refs_rows, _refs_by_paper = extract_reference_tables(chunks)
    in_corpus_cites = match_in_corpus_citations(refs_rows, df_papers)
    citations = citation_graph_metrics(in_corpus_cites, df_papers)

    claims_section = claim_controversy_and_maturity(df_papers, paper_rows)
    edge_controversy = claim_edge_controversy_metrics(claim_edge_snapshot, df_papers)
    stress_gaps = stress_test_gap_analytics(user_events)
    if not isinstance(claims_section, dict):
        claims_section = {"available": False, "reason": "invalid claims section"}
    claims_section["edge_controversy"] = edge_controversy
    claims_section["stress_test_gaps"] = stress_gaps
    claims_section["controversy_method"] = "claim_edges" if edge_controversy.get("available") else "text_heuristic"
    if edge_controversy.get("available") and edge_controversy.get("per_cluster"):
        # Keep legacy key name so Swift can decode without changes.
        claims_section["controversy_by_cluster"] = edge_controversy.get("per_cluster", [])
        claims_section["available"] = True
    elif stress_gaps.get("available"):
        claims_section["available"] = True
    methods_section = methods_datasets_adoption(df_papers, chunks)
    ingestion_diag = ingestion_quality_diagnostics(df_papers, chunks)
    duplicates = detect_duplicates(df_papers, Z_whitened, knn)

    workflow = workflow_coverage_and_blindspots(df_papers, Z_whitened, user_events, cluster_centroids_z)
    qa_gaps = qa_gap_analytics(user_events, df_papers)
    mig = marginal_information_gain_recommendations(df_papers, Z_whitened, cluster_centroids_z, workflow, n_select=8)

    # Primary recommendations: MIG (fallback to simple heuristic)
    recs_mig = [row["paper_id"] for row in mig.get("selected", [])] if mig.get("available") else []
    novelty_for_recs = [{"paper_id": pid, "novelty": vals.get("nov_cluster", 0.0)} for pid, vals in multi_nov.items()]
    recs_simple = recommend_papers(loadings, novelty_for_recs, centrality, k=6)
    recs = recs_mig if recs_mig else recs_simple

    # Uncertainty proxy using bootstrap of kNN sets (reuse knn list)
    top_sets = []
    for i in range(min(5, len(knn))):
        neighbors = knn[i].get("neighbors", [])
        top_sets.append([n["paper_id"] for n in neighbors[:6]])
    confidence = estimate_answer_confidence(top_sets)

    event_stats = {
        "total": len(user_events),
        "by_type": {},
        "last_seen": None,
    }
    if user_events:
        counts: dict[str, int] = {}
        last_ts: str | None = None
        for evt in user_events:
            etype = evt.get("event_type", "unknown")
            counts[etype] = counts.get(etype, 0) + 1
            ts = evt.get("timestamp")
            if isinstance(ts, str):
                if last_ts is None or ts > last_ts:
                    last_ts = ts
        event_stats["by_type"] = counts
        event_stats["last_seen"] = last_ts

    # Counterfactual scenarios (year cutoffs)
    counterfactuals = []
    for cutoff in args.counterfactual_cutoffs:
        mask = df_papers.year.fillna(0) >= cutoff
        kept = df_papers[mask]
        kept_ids = set(kept.paper_id)
        cent_vals = [c["weighted_degree"] for c in centrality if c["paper_id"] in kept_ids]
        avg_c = float(sum(cent_vals) / len(cent_vals)) if cent_vals else 0.0
        counterfactuals.append({"name": f"year>={cutoff}", "paper_count": int(mask.sum()), "avg_centrality": avg_c})

    # Drift contribution + paper metrics
    influence_abs_map = {item["paper_id"]: item["influence"] for item in influence_abs}
    drift_contrib = drift_contribution(df_papers, Z_whitened, drift_vec_map, influence_abs_map)
    paper_metrics = build_paper_metrics(
        df_papers=df_papers,
        multi_novelty=multi_nov,
        comb_novelty=comb_nov,
        novelty_unc=nov_unc,
        centrality_deg={c["paper_id"]: c["weighted_degree"] for c in centrality},
        eigen_cent=eigen_cent,
        edges=claim_edges,
        influence_abs=influence_abs,
        influence_pos=influence_pos,
        influence_neg=influence_neg,
        out_strength=out_strength,
        in_strength=in_strength,
        drift_contrib_map=drift_contrib,
    )

    # Merge additional per-paper metrics into paper_metrics for convenience on the Swift side.
    metrics_by_id = {m["paper_id"]: m for m in paper_metrics}
    cluster_by_pid = {str(r.paper_id): (int(r.cluster_id) if r.cluster_id is not None and not pd.isna(r.cluster_id) else None) for _, r in df_papers.iterrows()}

    if stability.get("available"):
        for row in stability.get("per_paper", []):
            pid = str(row.get("paper_id"))
            if pid in metrics_by_id:
                metrics_by_id[pid]["cluster_confidence"] = float(row.get("cluster_confidence", 0.0))
                metrics_by_id[pid]["cluster_ambiguity"] = float(row.get("ambiguity", 0.0))
                metrics_by_id[pid]["cluster_top1"] = row.get("top1_cluster")
                metrics_by_id[pid]["cluster_top2"] = row.get("top2_cluster")
                metrics_by_id[pid]["cluster_margin"] = float(row.get("margin", 0.0))
                metrics_by_id[pid]["cluster_dist_margin"] = float(row.get("dist_margin", 0.0))

    if recombination.get("available"):
        for row in recombination.get("per_paper", []):
            pid = str(row.get("paper_id"))
            if pid in metrics_by_id:
                metrics_by_id[pid]["recombination"] = float(row.get("recombination", 0.0))

    if methods_section.get("available"):
        for row in methods_section.get("per_paper", []):
            pid = str(row.get("paper_id"))
            if pid in metrics_by_id:
                metrics_by_id[pid]["rigor_proxy"] = float(row.get("rigor_proxy", 0.0))
                metrics_by_id[pid]["openness_score"] = float(row.get("openness_score", 0.0))
                metrics_by_id[pid]["has_code_link"] = bool(row.get("has_code_link", False))
                metrics_by_id[pid]["has_data_link"] = bool(row.get("has_data_link", False))

    if citations.get("available"):
        for row in citations.get("pagerank", []):
            pid = str(row.get("paper_id"))
            if pid in metrics_by_id:
                metrics_by_id[pid]["citation_pagerank"] = float(row.get("pagerank", 0.0))
                metrics_by_id[pid]["citation_in_degree"] = int(row.get("in_degree", 0))

    if workflow.get("available"):
        read_map = {str(r.get("paper_id")): float(r.get("read_score", 0.0)) for r in workflow.get("paper_read_scores", [])}
        for pid, sc in read_map.items():
            if pid in metrics_by_id:
                metrics_by_id[pid]["read_score"] = float(sc)

    if map_quality.get("available"):
        dist_by_cluster = {int(r["cluster_id"]): float(r.get("distortion", 0.0)) for r in map_quality.get("local_distortion", []) if r.get("cluster_id") is not None}
        for pid, m in metrics_by_id.items():
            cid = cluster_by_pid.get(pid)
            if cid is not None:
                m["layout_distortion"] = float(dist_by_cluster.get(int(cid), 0.0))

    if paper_map_quality.get("available") and paper_map_distortion:
        for pid, dist in paper_map_distortion.items():
            if pid in metrics_by_id:
                metrics_by_id[pid]["paper_layout_distortion"] = float(dist)

    if duplicates.get("available"):
        group_for: dict[str, int] = {}
        for gi, group in enumerate(duplicates.get("groups", [])[:200]):
            for pid in group:
                group_for[str(pid)] = gi
        for pid, gi in group_for.items():
            if pid in metrics_by_id:
                metrics_by_id[pid]["duplicate_group"] = int(gi)

    if ingestion_diag.get("available"):
        flags_for = {str(r.get("paper_id")): r.get("flags", []) for r in ingestion_diag.get("issues", [])}
        for pid, flags in flags_for.items():
            if pid in metrics_by_id and isinstance(flags, list):
                metrics_by_id[pid]["ingestion_flags"] = flags
    paper_metrics = list(metrics_by_id.values())

    # Export whitened embeddings parquet for notebooks/Swift if needed
    emb_whitened_path = paths["analytics_dir"] / "embeddings_whitened.parquet"
    pd.DataFrame({"paper_id": df_papers.paper_id, "embedding_z": list(Z_whitened)}).to_parquet(emb_whitened_path, compression="zstd")

    # Prepare novelty list (top outliers by cluster distance)
    novelty_sorted = sorted(
        (
            {
                "paper_id": pid,
                "cluster_id": int(df_papers.loc[df_papers.paper_id == pid, "cluster_id"].iloc[0]) if not pd.isna(df_papers.loc[df_papers.paper_id == pid, "cluster_id"].iloc[0]) else None,
                "novelty": vals.get("nov_cluster", 0.0),
            }
            for pid, vals in multi_nov.items()
        ),
        key=lambda x: x["novelty"],
        reverse=True,
    )

    # Persist DuckDB after computing extra tables (optional)
    extra_tables: dict[str, pd.DataFrame] = {}
    if refs_rows:
        extra_tables["refs"] = pd.DataFrame(refs_rows)
    if in_corpus_cites:
        extra_tables["in_corpus_cites"] = pd.DataFrame(in_corpus_cites)
    if methods_section.get("available"):
        ents_df = pd.DataFrame(
            [
                {
                    "paper_id": r.get("paper_id"),
                    "methods_json": json.dumps(r.get("methods", [])),
                    "datasets_json": json.dumps(r.get("datasets", [])),
                    "metrics_json": json.dumps(r.get("metrics", [])),
                    "rigor_proxy": float(r.get("rigor_proxy", 0.0)),
                    "openness_score": float(r.get("openness_score", 0.0)),
                    "has_code_link": bool(r.get("has_code_link", False)),
                    "has_data_link": bool(r.get("has_data_link", False)),
                    "artifact_links_json": json.dumps(r.get("artifact_links", [])),
                }
                for r in methods_section.get("per_paper", [])[:5000]
            ]
        )
        extra_tables["paper_entities"] = ents_df
    if duplicates.get("available"):
        extra_tables["duplicate_pairs"] = pd.DataFrame(duplicates.get("pairs", []))
    if ingestion_diag.get("available"):
        extra_tables["ingestion_issues"] = pd.DataFrame(
            [
                {
                    "paper_id": r.get("paper_id"),
                    "flags_json": json.dumps(r.get("flags", [])),
                    "chunk_text_len": r.get("chunk_text_len"),
                    "page_count": r.get("page_count"),
                }
                for r in ingestion_diag.get("issues", [])[:5000]
            ]
        )
    if qa_gaps.get("available"):
        extra_tables["qa_questions"] = pd.DataFrame(qa_gaps.get("questions", []))
    if topic_graph.get("available"):
        extra_tables["topic_graph_edges"] = pd.DataFrame(topic_graph.get("edges", []))
    if claim_edge_snapshot and isinstance(claim_edge_snapshot.get("edges"), list):
        gen_at = claim_edge_snapshot.get("generatedAt") or claim_edge_snapshot.get("generated_at")
        edge_rows: list[dict[str, Any]] = []
        for e in claim_edge_snapshot.get("edges", [])[:500000]:
            if not isinstance(e, dict):
                continue
            edge_rows.append(
                {
                    "generated_at": gen_at,
                    "src_paper_id": e.get("srcPaperID") or e.get("src_paper_id"),
                    "dst_paper_id": e.get("dstPaperID") or e.get("dst_paper_id"),
                    "source_claim_id": e.get("sourceClaimID") or e.get("source_claim_id"),
                    "target_claim_id": e.get("targetClaimID") or e.get("target_claim_id"),
                    "kind": e.get("kind"),
                    "rationale": e.get("rationale"),
                }
            )
        if edge_rows:
            extra_tables["claim_edges"] = pd.DataFrame(edge_rows)

    stress_events = [e for e in (user_events or []) if isinstance(e, dict) and e.get("event_type") == "assumption_stress_test"]
    if stress_events:
        stress_rows: list[dict[str, Any]] = []
        for evt in stress_events[-10000:]:
            ts = evt.get("timestamp")
            assumption = evt.get("assumption")
            clusters = evt.get("clusters") or {}
            if not isinstance(clusters, dict):
                clusters = {}
            suggested = evt.get("suggested_tests") or []
            suggested_json = json.dumps(suggested if isinstance(suggested, list) else [])
            for k, v in clusters.items():
                try:
                    cid = int(k)
                except Exception:
                    continue
                try:
                    count = int(v)
                except Exception:
                    count = 0
                stress_rows.append(
                    {
                        "timestamp": ts,
                        "assumption": assumption,
                        "cluster_id": cid,
                        "affected_papers_in_cluster": count,
                        "affected_claims": evt.get("affected_claims"),
                        "affected_papers": evt.get("affected_papers"),
                        "suggested_tests_json": suggested_json,
                    }
                )
        if stress_rows:
            extra_tables["stress_tests"] = pd.DataFrame(stress_rows)

    persist_duckdb(db_path, df_papers, embeddings, chunks, extra_tables=extra_tables)

    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paper_count": len(df_papers),
        "vector_dim": int(embeddings.shape[1]),
        "topic_trends": topic_trends,
        "novelty": novelty_sorted[:50],  # top outliers
        "centrality": centrality[:200],
        "drift": drift,
        "drift_volatility": drift_vol,
        "factors": factors,
        "factor_loadings": loadings[:500],
        "factor_exposures": factor_exposures,
        "factor_exposures_user": factor_exposures_user,
        "factor_labels": factor_labels,
        "influence": influence_abs,
        "influence_pos": influence_pos,
        "influence_neg": influence_neg,
        "idea_flow_edges": claim_edges,
        "paper_metrics": paper_metrics,
        "quality": {
            "map": map_quality,
            "paper_map": paper_map_quality,
            "ingestion": ingestion_diag,
        },
        "stability": stability,
        "stability_boundary": stability_boundary,
        "lifecycle": lifecycle,
        "bridges": {
            "topic_graph": topic_graph,
            "topic_metrics": topic_metrics[:400],
            "paper_recombination": recombination,
        },
        "citations": {
            "refs_extracted": int(len(refs_rows)),
            "in_corpus_cites": int(len(in_corpus_cites)),
            "graph": citations,
        },
        "claims": claims_section,
        "methods": methods_section,
        "workflow": {
            "coverage": workflow,
            "qa_gaps": qa_gaps,
            "recommendations_mig": mig,
        },
        "hygiene": {
            "duplicates": duplicates,
        },
        "recommendations": recs,
        "recommendations_simple": recs_simple,
        "answer_confidence": confidence,
        "counterfactuals": counterfactuals,
        "user_events": event_stats,
        "paths": {
            "duckdb": str(db_path),
            "parquet_dir": str(paths["analytics_dir"]),
            "embeddings_whitened": str(emb_whitened_path),
            "galaxy_version": str(galaxy_layout.get("version")) if galaxy_layout else None,
        },
        "notes": (
            "Top-50 novelty list is sorted descending (higher = farther from cluster centroid). "
            "Centrality uses weighted degree over 8-NN cosine graph. user_events is populated if analytics/user_events.jsonl exists. "
            "paper_metrics carries z-scored novelty/consensus, influence decomposition, drift contribution, and roles; extended with stability, rigor, openness, citations, and workflow signals when available."
        ),
    }
    write_summary(paths["analytics_dir"], summary)


if __name__ == "__main__":
    main()
