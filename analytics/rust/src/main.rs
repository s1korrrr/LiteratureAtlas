use clap::Parser;
use polars::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Parquet file with columns paper_id (Utf8) and embedding (List<Float64 or Float32>)
    #[arg(long, default_value = "Output/analytics/paper_embeddings.parquet")]
    emb: PathBuf,
    /// Output JSON path for ANN edges
    #[arg(long, default_value = "Output/analytics/ann_edges.json")]
    out: PathBuf,
    /// Neighbors per node
    #[arg(long, default_value_t = 8)]
    k: usize,
}

#[derive(Serialize)]
struct Neighbor {
    paper_id: String,
    score: f64,
}

#[derive(Serialize)]
struct AnnEntry {
    paper_id: String,
    neighbors: Vec<Neighbor>,
    weighted_degree: f64,
    average_similarity: f64,
}

fn main() -> polars::prelude::PolarsResult<()> {
    let args = Args::parse();

    let lf = LazyFrame::scan_parquet(args.emb.to_string_lossy().as_ref(), Default::default())?;
    let df = lf.collect()?;

    let paper_ids = df
        .column("paper_id")?
        .str()?
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let embeddings_col = df.column("embedding")?;

    let mut vectors: Vec<Vec<f64>> = Vec::with_capacity(embeddings_col.len());
    for i in 0..embeddings_col.len() {
        let vals: Vec<f64> = match embeddings_col.get(i)? {
            AnyValue::List(series) => match series.dtype() {
                DataType::Float64 => series.f64()?.into_no_null_iter().collect(),
                DataType::Float32 => series
                    .f32()?
                    .into_no_null_iter()
                    .map(|v| v as f64)
                    .collect(),
                other => {
                    return Err(PolarsError::ComputeError(
                        format!("Unsupported embedding dtype: {other:?}").into(),
                    ))
                }
            },
            other => {
                return Err(PolarsError::ComputeError(
                    format!("Row {i} has non-list embedding: {other:?}").into(),
                ))
            }
        };
        vectors.push(vals);
    }

    if paper_ids.len() != vectors.len() {
        return Err(PolarsError::ComputeError(
            "paper_id and embedding lengths differ".into(),
        ));
    }

    let n = vectors.len();
    if n == 0 {
        println!("No embeddings found; exiting.");
        return Ok(());
    }

    // Normalize
    let mut normed: Vec<Vec<f64>> = Vec::with_capacity(n);
    for v in &vectors {
        let norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt().max(1e-12);
        normed.push(v.iter().map(|x| x / norm).collect());
    }

    let mut entries: Vec<AnnEntry> = Vec::with_capacity(n);
    for i in 0..n {
        let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i == j {
                continue;
            }
            let s = dot(&normed[i], &normed[j]);
            scores.push((j, s));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(args.k);
        let weighted_degree: f64 = scores.iter().map(|(_, s)| s).sum();
        let avg = if scores.is_empty() {
            0.0
        } else {
            weighted_degree / scores.len() as f64
        };
        let neighbors = scores
            .iter()
            .map(|(j, s)| Neighbor {
                paper_id: paper_ids[*j].to_string(),
                score: *s,
            })
            .collect();
        entries.push(AnnEntry {
            paper_id: paper_ids[i].to_string(),
            neighbors,
            weighted_degree,
            average_similarity: avg,
        });
    }

    let mut file = File::create(&args.out)?;
    serde_json::to_writer_pretty(&mut file, &entries).expect("write json");
    file.flush().ok();
    println!(
        "[atlas_graph] wrote {} entries to {}",
        entries.len(),
        args.out.display()
    );
    Ok(())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
