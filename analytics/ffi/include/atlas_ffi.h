#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t index;
    float distance;
} AtlasSearchResult;

typedef struct {
    uint32_t src;
    uint32_t dst;
    float weight;
} AtlasEdge;

// Build HNSW index from a contiguous [n * dim] row-major float array.
void* atlas_build_index(uint32_t dim, uint32_t n, const float* data_ptr);

// Free index allocated by atlas_build_index.
void atlas_free_index(void* index_ptr);

// Query top-k neighbors. Returns number of results written to out_ptr.
uint32_t atlas_query_index(void* index_ptr, const float* query_ptr, uint32_t k, AtlasSearchResult* out_ptr);

// Betweenness centrality on an undirected weighted graph. out_ptr must hold n_nodes floats.
uint32_t atlas_betweenness(uint32_t n_nodes, uint32_t n_edges, const AtlasEdge* edges_ptr, float* out_ptr);

// Connected components; out_ptr must hold n_nodes u32 ids; returns component count.
uint32_t atlas_connected_components(uint32_t n_nodes, uint32_t n_edges, const AtlasEdge* edges_ptr, uint32_t* out_ptr);

#ifdef __cplusplus
}
#endif
