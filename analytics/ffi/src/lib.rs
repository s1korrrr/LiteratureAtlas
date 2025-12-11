use std::ptr;
use std::slice;

use hnsw_rs::prelude::{DistL2, Hnsw};
use petgraph::algo::connected_components;
use petgraph::graph::Graph;
use petgraph::visit::EdgeRef;
use petgraph::Undirected;

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
    #[allow(dead_code)]
    points: Vec<Vec<f32>>, // owns data so HNSW references stay valid
    index: Hnsw<'static, f32, DistL2>,
}

unsafe fn wrap_slice<'a>(ptr: *const f32, len: usize) -> &'a [f32] {
    slice::from_raw_parts(ptr, len)
}

#[no_mangle]
pub extern "C" fn atlas_build_index(dim: u32, n: u32, data_ptr: *const f32) -> *mut AtlasIndex {
    if data_ptr.is_null() || dim == 0 || n == 0 {
        return ptr::null_mut();
    }
    let dim_usize = dim as usize;
    let n_usize = n as usize;
    let slice = unsafe { wrap_slice(data_ptr, dim_usize * n_usize) };

    let mut points: Vec<Vec<f32>> = Vec::with_capacity(n_usize);
    for chunk in slice.chunks(dim_usize) {
        points.push(chunk.to_vec());
    }

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

    // Insert while keeping points alive in AtlasIndex
    for (idx, vec) in points.iter().enumerate() {
        index.insert((&vec[..], idx));
    }

    Box::into_raw(Box::new(AtlasIndex { dim: dim_usize, points, index }))
}

#[no_mangle]
pub extern "C" fn atlas_free_index(index_ptr: *mut AtlasIndex) {
    if !index_ptr.is_null() {
        unsafe { drop(Box::from_raw(index_ptr)); }
    }
}

#[no_mangle]
pub extern "C" fn atlas_query_index(
    index_ptr: *mut AtlasIndex,
    query_ptr: *const f32,
    k: u32,
    out_ptr: *mut AtlasSearchResult,
) -> u32 {
    if index_ptr.is_null() || query_ptr.is_null() || out_ptr.is_null() || k == 0 {
        return 0;
    }
    let idx = unsafe { &*index_ptr };
    let q = unsafe { wrap_slice(query_ptr, idx.dim) };

    let results = idx.index.search(q, k as usize, 64);
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, k as usize) };
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
pub extern "C" fn atlas_betweenness(
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
    let edges = unsafe { slice::from_raw_parts(edges_ptr, ne) };

    let mut g: Graph<(), f32, Undirected> = Graph::new_undirected();
    let nodes: Vec<_> = (0..nn).map(|_| g.add_node(())).collect();
    for e in edges {
        let s = e.src as usize;
        let d = e.dst as usize;
        if s < nn && d < nn {
            g.add_edge(nodes[s], nodes[d], e.weight);
        }
    }

    // Degree centrality approximation
    let mut centrality = vec![0.0f32; nn];
    for e in g.edge_references() {
        let (s, d) = (e.source().index(), e.target().index());
        let w = *e.weight();
        centrality[s] += w;
        centrality[d] += w;
    }

    let out = unsafe { slice::from_raw_parts_mut(out_ptr, nn) };
    out.copy_from_slice(&centrality);
    1
}

#[no_mangle]
pub extern "C" fn atlas_connected_components(
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
    let edges = unsafe { slice::from_raw_parts(edges_ptr, ne) };

    let mut g: Graph<(), (), Undirected> = Graph::new_undirected();
    let nodes: Vec<_> = (0..nn).map(|_| g.add_node(())).collect();
    for e in edges {
        let s = e.src as usize;
        let d = e.dst as usize;
        if s < nn && d < nn {
            g.add_edge(nodes[s], nodes[d], ());
        }
    }

    let comps = connected_components(&g);
    let sccs = petgraph::algo::kosaraju_scc(&g);
    let mut uf = vec![u32::MAX; nn];
    for (cid, comp) in sccs.iter().enumerate() {
        for node in comp {
            uf[node.index()] = cid as u32;
        }
    }

    let out = unsafe { slice::from_raw_parts_mut(out_ptr, nn) };
    out.copy_from_slice(&uf);
    comps as u32
}
