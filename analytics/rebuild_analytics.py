#!/usr/bin/env python3
"""
Rebuild the local analytics warehouse for LiteratureAtlas.

Pipeline:
- Read *.paper.json and chunks.json produced by the Swift app (Output/papers, Output/chunks).
- Load into DuckDB (atlas.duckdb) with typed tables for papers, embeddings, chunks, claims, methods.
- Compute lightweight analytics (topic trends, novelty) and export JSON + Parquet into Output/analytics/.

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
from dataclasses import dataclass
from typing import List, Dict, Any

import duckdb  # pip install duckdb>=1.0.0
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

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

# ---------- Helpers ----------

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
    k = min(max_components, embeddings.shape[1])
    pca = PCA(n_components=k, whiten=True, random_state=0)
    Z = pca.fit_transform(embeddings)
    return Z.astype(np.float32), pca

# ---------- Analytics ----------

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

def persist_duckdb(db_path: pathlib.Path, df_papers: pd.DataFrame, embeddings: np.ndarray, chunks: list[dict[str, Any]]):
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

    # Whiten embeddings for isotropic distance metrics
    Z_whitened, pca_model = whiten_embeddings(embeddings)

    print(f"[analytics] Loaded {len(df_papers)} papers; building DuckDB at {db_path}")
    chunks = load_chunks(paths["chunks_path"])
    user_events = load_user_events(paths["output"])
    persist_duckdb(db_path, df_papers, embeddings, chunks)

    topic_trends = compute_topic_trends(df_papers)

    drift = compute_drift(df_papers, Z_whitened)
    drift_vol = drift_volatility(drift)
    drift_vec_map = drift_vector_map(df_papers, Z_whitened)

    multi_nov = compute_multi_novelty(df_papers, Z_whitened, drift_vec_map)
    nov_unc = novelty_uncertainty(df_papers, Z_whitened, compute_cluster_centroids(df_papers, Z_whitened))
    comb_nov = combinational_novelty(df_papers)

    knn = build_knn(Z_whitened, df_papers.paper_id.tolist(), k=8)
    centrality = centrality_from_knn(knn)
    eigen_cent = eigen_centrality_from_knn(knn)

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

    # Simple recs
    novelty_for_recs = [{"paper_id": pid, "novelty": vals.get("nov_cluster", 0.0)} for pid, vals in multi_nov.items()]
    recs = recommend_papers(loadings, novelty_for_recs, centrality, k=6)

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

    summary = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
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
        "recommendations": recs,
        "answer_confidence": confidence,
        "counterfactuals": counterfactuals,
        "user_events": event_stats,
        "paths": {
            "duckdb": str(db_path),
            "parquet_dir": str(paths["analytics_dir"]),
            "embeddings_whitened": str(emb_whitened_path),
        },
        "notes": (
            "Top-50 novelty list is sorted descending (higher = farther from cluster centroid). "
            "Centrality uses weighted degree over 8-NN cosine graph. user_events is populated if analytics/user_events.jsonl exists. "
            "paper_metrics carries z-scored novelty/consensus, influence decomposition, drift contribution, and roles."
        ),
    }
    write_summary(paths["analytics_dir"], summary)


if __name__ == "__main__":
    main()
