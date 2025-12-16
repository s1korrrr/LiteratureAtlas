use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ptr;
use std::slice;

use hnsw_rs::prelude::{DistL2, Hnsw};

#[repr(C)]
pub struct AtlasSearchResult {
    pub index: u32,
    pub distance: f32,
}

#[repr(C)]
pub struct AtlasEdge {
    pub src: u32,
    pub dst: u32,
    pub weight: f32,
}

pub struct AtlasIndex {
    dim: usize,
    index: Hnsw<'static, f32, DistL2>,
}

unsafe fn wrap_slice<'a>(ptr: *const f32, len: usize) -> &'a [f32] {
    slice::from_raw_parts(ptr, len)
}

fn build_adjacency(n_nodes: usize, edges: &[AtlasEdge]) -> Vec<Vec<(usize, f64)>> {
    let mut adjacency: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_nodes];
    for edge in edges {
        let src = edge.src as usize;
        let dst = edge.dst as usize;
        if src >= n_nodes || dst >= n_nodes {
            continue;
        }
        let weight = edge.weight as f64;
        if !weight.is_finite() || weight < 0.0 {
            continue;
        }
        adjacency[src].push((dst, weight));
        adjacency[dst].push((src, weight));
    }
    adjacency
}

#[derive(Copy, Clone, Debug)]
struct HeapState {
    cost: f64,
    position: usize,
}

impl Eq for HeapState {}

impl PartialEq for HeapState {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position && self.cost == other.cost
    }
}

impl Ord for HeapState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior.
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.position.cmp(&other.position))
    }
}

impl PartialOrd for HeapState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[no_mangle]
/// Build an HNSW index from a contiguous row-major `[n * dim]` float array.
///
/// # Safety
/// - `data_ptr` must be non-null and point to at least `dim * n` contiguous `f32` values.
/// - The memory behind `data_ptr` must be readable for the duration of the call.
/// - The returned pointer (when non-null) must be released exactly once via `atlas_free_index`.
pub unsafe extern "C" fn atlas_build_index(
    dim: u32,
    n: u32,
    data_ptr: *const f32,
) -> *mut AtlasIndex {
    if data_ptr.is_null() || dim == 0 || n == 0 {
        return ptr::null_mut();
    }
    let dim_usize = dim as usize;
    let n_usize = n as usize;
    let slice = wrap_slice(data_ptr, dim_usize * n_usize);

    // Build HNSW: max_nb_connection, max_elements, max_layer, ef_construction, dist
    let max_layer = 16usize;
    let ef_construction = 200usize;
    let max_nb_connection = 32usize;
    let max_elements = n_usize.max(1);
    let index: Hnsw<'static, f32, DistL2> = Hnsw::new(
        max_nb_connection,
        max_elements,
        max_layer,
        ef_construction,
        DistL2 {},
    );

    for (idx, chunk) in slice.chunks(dim_usize).enumerate() {
        index.insert((chunk, idx));
    }

    Box::into_raw(Box::new(AtlasIndex {
        dim: dim_usize,
        index,
    }))
}

#[no_mangle]
/// Free an index created by `atlas_build_index`.
///
/// # Safety
/// - `index_ptr` must be null, or a pointer returned by `atlas_build_index` that has not already been freed.
/// - After this call, `index_ptr` must not be used again.
pub unsafe extern "C" fn atlas_free_index(index_ptr: *mut AtlasIndex) {
    if !index_ptr.is_null() {
        drop(Box::from_raw(index_ptr));
    }
}

#[no_mangle]
/// Query the index for `k` nearest neighbors.
///
/// # Safety
/// - `index_ptr` must be null, or a valid pointer returned by `atlas_build_index`.
/// - `query_ptr` must be non-null and point to at least `dim` floats, where `dim` matches the index dimension.
/// - `out_ptr` must be non-null and point to at least `k` writable `AtlasSearchResult` entries.
pub unsafe extern "C" fn atlas_query_index(
    index_ptr: *mut AtlasIndex,
    query_ptr: *const f32,
    k: u32,
    out_ptr: *mut AtlasSearchResult,
) -> u32 {
    if index_ptr.is_null() || query_ptr.is_null() || out_ptr.is_null() || k == 0 {
        return 0;
    }
    let idx = &*index_ptr;
    let q = wrap_slice(query_ptr, idx.dim);

    let results = idx.index.search(q, k as usize, 64);
    let out = slice::from_raw_parts_mut(out_ptr, k as usize);
    let count = results.len().min(k as usize);
    for i in 0..count {
        let r = &results[i];
        out[i] = AtlasSearchResult {
            index: r.d_id as u32,
            distance: r.distance as f32,
        };
    }
    count as u32
}

#[no_mangle]
/// Compute betweenness centrality on an undirected weighted graph.
///
/// Edge weights are interpreted as non-negative costs (lower cost = shorter path). Invalid edges
/// (out-of-range node ids, non-finite weights, or negative weights) are ignored.
///
/// # Safety
/// - `edges_ptr` must be non-null and point to at least `n_edges` readable `AtlasEdge` values.
/// - `out_ptr` must be non-null and point to at least `n_nodes` writable `f32` values.
pub unsafe extern "C" fn atlas_betweenness(
    n_nodes: u32,
    n_edges: u32,
    edges_ptr: *const AtlasEdge,
    out_ptr: *mut f32,
) -> u32 {
    if edges_ptr.is_null() || out_ptr.is_null() {
        return 0;
    }
    let nn = n_nodes as usize;
    let ne = n_edges as usize;
    let edges = slice::from_raw_parts(edges_ptr, ne);

    if nn == 0 {
        return 0;
    }

    let adjacency = build_adjacency(nn, edges);

    // Brandes' algorithm (weighted) for betweenness centrality.
    let mut centrality = vec![0.0f64; nn];
    let eps = 1e-12f64;

    for s in 0..nn {
        let mut stack: Vec<usize> = Vec::with_capacity(nn);
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); nn];
        let mut sigma: Vec<f64> = vec![0.0; nn];
        let mut dist: Vec<f64> = vec![f64::INFINITY; nn];

        sigma[s] = 1.0;
        dist[s] = 0.0;

        let mut heap: BinaryHeap<HeapState> = BinaryHeap::new();
        heap.push(HeapState {
            cost: 0.0,
            position: s,
        });

        while let Some(HeapState { cost, position: v }) = heap.pop() {
            if cost > dist[v] + eps {
                continue;
            }
            stack.push(v);

            for &(w, weight) in &adjacency[v] {
                let vw_dist = dist[v] + weight;
                if vw_dist + eps < dist[w] {
                    dist[w] = vw_dist;
                    heap.push(HeapState {
                        cost: vw_dist,
                        position: w,
                    });
                    sigma[w] = sigma[v];
                    predecessors[w].clear();
                    predecessors[w].push(v);
                } else if (vw_dist - dist[w]).abs() <= eps {
                    sigma[w] += sigma[v];
                    predecessors[w].push(v);
                }
            }
        }

        let mut delta: Vec<f64> = vec![0.0; nn];
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w] {
                if sigma[w] != 0.0 {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != s {
                centrality[w] += delta[w];
            }
        }
    }

    // Undirected graphs are double-counted.
    for c in &mut centrality {
        *c *= 0.5;
    }

    let out = slice::from_raw_parts_mut(out_ptr, nn);
    for i in 0..nn {
        out[i] = centrality[i] as f32;
    }
    1
}

#[no_mangle]
/// Compute connected components on an undirected graph.
///
/// # Safety
/// - `edges_ptr` must be non-null and point to at least `n_edges` readable `AtlasEdge` values.
/// - `out_ptr` must be non-null and point to at least `n_nodes` writable `u32` values.
pub unsafe extern "C" fn atlas_connected_components(
    n_nodes: u32,
    n_edges: u32,
    edges_ptr: *const AtlasEdge,
    out_ptr: *mut u32,
) -> u32 {
    if edges_ptr.is_null() || out_ptr.is_null() {
        return 0;
    }
    let nn = n_nodes as usize;
    let ne = n_edges as usize;
    let edges = slice::from_raw_parts(edges_ptr, ne);

    if nn == 0 {
        return 0;
    }

    let adjacency = build_adjacency(nn, edges);

    let mut component_ids: Vec<u32> = vec![u32::MAX; nn];
    let mut visited: Vec<bool> = vec![false; nn];
    let mut component_count: u32 = 0;

    for start in 0..nn {
        if visited[start] {
            continue;
        }

        let mut stack = vec![start];
        visited[start] = true;
        component_ids[start] = component_count;

        while let Some(v) = stack.pop() {
            for &(w, _) in &adjacency[v] {
                if !visited[w] {
                    visited[w] = true;
                    component_ids[w] = component_count;
                    stack.push(w);
                }
            }
        }

        component_count += 1;
    }

    let out = slice::from_raw_parts_mut(out_ptr, nn);
    out.copy_from_slice(&component_ids);
    component_count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx_eq(actual: f32, expected: f32, eps: f32) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= eps,
            "expected {expected} got {actual} (diff {diff} > {eps})"
        );
    }

    #[test]
    fn connected_components_two_components() {
        let edges = vec![
            AtlasEdge {
                src: 0,
                dst: 1,
                weight: 1.0,
            },
            AtlasEdge {
                src: 2,
                dst: 3,
                weight: 1.0,
            },
        ];
        let mut out = vec![0u32; 4];
        let comps = unsafe {
            atlas_connected_components(4, edges.len() as u32, edges.as_ptr(), out.as_mut_ptr())
        };
        assert_eq!(comps, 2);
        assert_eq!(out[0], out[1]);
        assert_eq!(out[2], out[3]);
        assert_ne!(out[0], out[2]);
    }

    #[test]
    fn betweenness_line_graph() {
        let edges = vec![
            AtlasEdge {
                src: 0,
                dst: 1,
                weight: 1.0,
            },
            AtlasEdge {
                src: 1,
                dst: 2,
                weight: 1.0,
            },
            AtlasEdge {
                src: 2,
                dst: 3,
                weight: 1.0,
            },
        ];
        let mut out = vec![0f32; 4];
        let ok =
            unsafe { atlas_betweenness(4, edges.len() as u32, edges.as_ptr(), out.as_mut_ptr()) };
        assert_eq!(ok, 1);
        assert_approx_eq(out[0], 0.0, 1e-5);
        assert_approx_eq(out[1], 2.0, 1e-5);
        assert_approx_eq(out[2], 2.0, 1e-5);
        assert_approx_eq(out[3], 0.0, 1e-5);
    }

    #[test]
    fn betweenness_triangle_is_zero() {
        let edges = vec![
            AtlasEdge {
                src: 0,
                dst: 1,
                weight: 1.0,
            },
            AtlasEdge {
                src: 1,
                dst: 2,
                weight: 1.0,
            },
            AtlasEdge {
                src: 2,
                dst: 0,
                weight: 1.0,
            },
        ];
        let mut out = vec![0f32; 3];
        let ok =
            unsafe { atlas_betweenness(3, edges.len() as u32, edges.as_ptr(), out.as_mut_ptr()) };
        assert_eq!(ok, 1);
        assert_approx_eq(out[0], 0.0, 1e-5);
        assert_approx_eq(out[1], 0.0, 1e-5);
        assert_approx_eq(out[2], 0.0, 1e-5);
    }
}
