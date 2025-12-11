# atlas_graph (Rust helper)

Small CLI that turns Parquet embeddings into a k-NN graph for downstream analytics.

## Usage
```bash
cargo run --manifest-path analytics/rust/Cargo.toml --release -- \
  --emb Output/analytics/paper_embeddings.parquet \
  --out Output/analytics/ann_edges.json \
  --k 8
```

## Flags
- `--emb`  : Parquet file containing `paper_id` (Utf8) and `embedding` (List<Float32/Float64>). Default: `Output/analytics/paper_embeddings.parquet`.
- `--out`  : Destination JSON path for ANN edges. Default: `Output/analytics/ann_edges.json`.
- `--k`    : Neighbors per node (default: 8).

## Output
`ann_edges.json` — array of entries:
- `paper_id`
- `neighbors` (paper_id, score)
- `weighted_degree`
- `average_similarity`

## Notes
- Uses brute-force cosine; suitable for small/medium corpora. Swap to HNSW if scale grows.
- Relies on Polars for Parquet I/O; first build will compile dependencies, subsequent runs are fast.
