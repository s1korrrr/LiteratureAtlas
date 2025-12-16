# Local Analytics Backend (Python + DuckDB + Rust helpers)

This directory houses the fully local analytics pipeline that the SwiftUI app can regenerate on demand. All reads/writes stay under `Output/`—no cloud dependencies.

## What it does
- Loads `Output/papers/*.paper.json` and `Output/chunks/chunks.json` produced by the app.
- Builds `Output/atlas.duckdb` with typed tables (papers, embeddings, chunks, claims, methods).
- Computes topic trends, novelty vs. cluster centroids, 8-NN centrality, drift, factor loadings/exposures, influence scores, recommendations, uncertainty proxy, counterfactual scenarios, and optional user-event stats from `Output/analytics/user_events.jsonl`.
- Adds higher-level offline analytics:
  - **Stability**: cluster assignment confidence/ambiguity + cohesion.
  - **Map quality**: trustworthiness/continuity + local distortion for the galaxy layout (cluster-level).
  - **Lifecycle**: burst scores, changepoint heuristics, phase labels (emerging/accelerating/mature/fading).
  - **Bridges**: topic graph betweenness/bridging-centrality + paper recombination index.
  - **Citations** (heuristic, offline): reference extraction + in-corpus matching + PageRank/in-degree when references are present in `chunks.json`.
  - **Claims**: controversy + maturity proxies from claim text similarity.
  - **Methods/datasets/metrics**: lightweight extraction + adoption curves + rigor/openness proxies.
  - **Workflow/hygiene**: coverage/blind-spots, QA gaps (from user events), duplicates + ingestion diagnostics.
- Exports Parquet snapshots and a compact `Output/analytics/analytics.json` consumed by `AnalyticsView` in the app.

## Quick start
```bash
# from repo root
python analytics/rebuild_analytics.py
# or if running from elsewhere
python analytics/rebuild_analytics.py --base /path/to/LiteratureAtlas
```

Key outputs
- `Output/atlas.duckdb` — warehouse (tables: papers, paper_embeddings, paper_chunks, claims, methods).
- `Output/analytics/*.parquet` — convenient parquet snapshots (papers, embeddings, chunks, claims, methods, plus optional extras like `paper_entities.parquet`, `ingestion_issues.parquet`, `refs.parquet`, `in_corpus_cites.parquet` when available).
- `Output/analytics/analytics.json` — summarized payload (baseline metrics plus new sections: `quality`, `stability`, `lifecycle`, `bridges`, `citations`, `claims`, `methods`, `workflow`, `hygiene`).

## Dependencies
```bash
pip install -r analytics/requirements.txt
# or with uv:
# uv venv && uv pip install -r analytics/requirements.txt
```
- duckdb>=1.0.0, numpy>=1.26, pandas>=2.0, scikit-learn>=1.5

## Script options (rebuild_analytics.py)
- `--base / --root` : repo root (default: script parent).
- `--db` : custom DuckDB path (default: `Output/atlas.duckdb`).
- `--counterfactual-cutoffs` : list of year thresholds for counterfactual scenarios (default: `2010 2015 2020`).

## Rust helpers
- **ANN graph CLI** (`analytics/rust`): builds k-NN edges from `paper_embeddings.parquet`.
  ```bash
  cargo run --manifest-path analytics/rust/Cargo.toml --release -- \
    --emb Output/analytics/paper_embeddings.parquet \
    --out Output/analytics/ann_edges.json \
    --k 8
  ```
  See `analytics/rust/README.md` for flags.

- **Swift FFI library** (`analytics/ffi`): exposes HNSW search and lightweight graph analytics.
  ```bash
  cargo build --manifest-path analytics/ffi/Cargo.toml --release
  # Headers: analytics/ffi/include/atlas_ffi.h
  # Library: analytics/ffi/target/release/libatlas_ffi.{dylib,a}
  ```
  The Swift target links against this library (see `Package.swift`).

## Notebooks & extension points
- Open `Output/atlas.duckdb` in Jupyter/duckdb for custom analyses (UMAP, graph stats, topic modeling). Keep heavy experiments here; only small artifacts should flow back into `analytics.json`.
- To extend metrics, write additional Parquet/JSON files into `Output/analytics/` and reference them from the app if needed.
