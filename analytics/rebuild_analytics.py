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


def compute_novelty(df_papers: pd.DataFrame, embeddings: np.ndarray) -> list[dict[str, Any]]:
    centroids = compute_cluster_centroids(df_papers, embeddings)
    results: list[dict[str, Any]] = []
    for i, row in df_papers.iterrows():
        emb = embeddings[i]
        cid = row.cluster_id
        if cid is None or cid not in centroids:
            continue
        sim = cosine_similarity(emb, centroids[int(cid)])
        novelty = max(0.0, 1.0 - ((sim + 1) / 2))  # map cosine [-1,1] to [0,1] distance
        results.append(
            {
                "paper_id": row.paper_id,
                "cluster_id": int(cid),
                "novelty": round(novelty, 6),
            }
        )
    # Sort descending novelty
    results.sort(key=lambda x: x["novelty"], reverse=True)
    return results


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

def compute_factor_loadings(embeddings: np.ndarray, paper_ids: list[str], n_factors: int = 8):
    # Center
    X = embeddings - embeddings.mean(axis=0, keepdims=True)
    # SVD
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    comps = vt[:n_factors]
    scores = np.dot(X, comps.T)
    loadings = []
    for pid, row in zip(paper_ids, scores):
        loadings.append({"paper_id": pid, "scores": row.tolist()})
    factors = comps.tolist()
    return factors, loadings


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


# ---------- Influence (simple PageRank) ----------

def pagerank(n: int, edges: list[tuple[int, int]], d: float = 0.85, iters: int = 40):
    if n == 0:
        return []
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    out_deg = np.zeros(n)
    for u, v in edges:
        out_deg[u] += 1
    for _ in range(iters):
        new = np.full(n, (1 - d) / n, dtype=np.float64)
        for u, v in edges:
            if out_deg[u] > 0:
                new[v] += d * rank[u] / out_deg[u]
        rank = new
    return rank.tolist()


def compute_influence(df_papers: pd.DataFrame, claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not claims:
        return []
    # Map paper_id to index
    ids = df_papers.paper_id.tolist()
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    edges: list[tuple[int, int]] = []
    for c in claims:
        pid = c.get("paper_id")
        year = c.get("year")
        # create edges from older to newer similar claims: heuristic via shared assumptions/keywords
        if pid not in id_to_idx:
            continue
        for other in claims:
            if other is c:
                continue
            if other.get("paper_id") not in id_to_idx:
                continue
            oy = other.get("year")
            if oy is None or year is None or oy >= year:
                continue
            if set(c.get("assumptions", [])) & set(other.get("assumptions", [])):
                edges.append((id_to_idx[other["paper_id"]], id_to_idx[pid]))
    ranks = pagerank(len(ids), edges)
    return [
        {"paper_id": pid, "influence": float(ranks[i])}
        for i, pid in enumerate(ids)
    ]


def idea_flow_edges(claims: list[dict[str, Any]], id_to_idx: dict[str, int]) -> list[dict[str, Any]]:
    edges = []
    for c in claims:
        pid = c.get("paper_id")
        year = c.get("year")
        for other in claims:
            if other is c:
                continue
            oy = other.get("year")
            if oy is None or year is None or oy >= year:
                continue
            if set(c.get("assumptions", [])) & set(other.get("assumptions", [])):
                edges.append({
                    "src": other.get("paper_id"),
                    "dst": pid,
                    "weight": 1.0,
                    "from_year": oy,
                    "to_year": year
                })
    return edges


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

    print(f"[analytics] Loaded {len(df_papers)} papers; building DuckDB at {db_path}")
    chunks = load_chunks(paths["chunks_path"])
    user_events = load_user_events(paths["output"])
    persist_duckdb(db_path, df_papers, embeddings, chunks)

    topic_trends = compute_topic_trends(df_papers)
    novelty = compute_novelty(df_papers, embeddings)
    knn = build_knn(embeddings, df_papers.paper_id.tolist(), k=8)
    centrality = centrality_from_knn(knn)
    drift = compute_drift(df_papers, embeddings)

    # Factor model
    factors, loadings = compute_factor_loadings(embeddings, df_papers.paper_id.tolist(), n_factors=8)
    factor_exposures = factor_exposures_over_time(loadings, df_papers.year.tolist(), n_factors=8)

    # Influence via claims
    all_claims: list[dict[str, Any]] = []
    for row in paper_rows:
        if row.claims:
            for c in row.claims:
                all_claims.append({
                    "paper_id": row.paper_id,
                    "assumptions": c.get("assumptions", []),
                    "year": c.get("year") or row.year,
                })
    influence = compute_influence(df_papers, all_claims)

    # Simple recs
    recs = recommend_papers(loadings, novelty, centrality, k=6)

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

    summary = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "paper_count": len(df_papers),
        "vector_dim": int(embeddings.shape[1]),
        "topic_trends": topic_trends,
        "novelty": novelty[:50],  # top outliers
        "centrality": centrality[:200],
        "drift": drift,
        "factors": factors,
        "factor_loadings": loadings[:500],
        "factor_exposures": factor_exposures,
        "influence": influence,
        "idea_flow_edges": idea_flow_edges(all_claims, {}),
        "recommendations": recs,
        "answer_confidence": confidence,
        "counterfactuals": counterfactuals,
        "user_events": event_stats,
        "paths": {
            "duckdb": str(db_path),
            "parquet_dir": str(paths["analytics_dir"]),
        },
        "notes": (
            "Top-50 novelty list is sorted descending (higher = farther from cluster centroid). "
            "Centrality uses weighted degree over 8-NN cosine graph. user_events is populated if analytics/user_events.jsonl exists."
        ),
    }
    write_summary(paths["analytics_dir"], summary)


if __name__ == "__main__":
    main()
