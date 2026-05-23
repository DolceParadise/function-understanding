# Embedding Evaluation Report

This report scores the demo output files produced by the retrieval and clustering scripts.
Retrieval uses label agreement as a semantic proxy, with both strict and soft matching variants.
For full Recall@K coverage, the retrieval demo should emit at least K ranked neighbors.
Clustering is summarized with purity and size-based sanity checks.

## Retrieval Evaluation

Query function: __default_allocate
Query purpose: allocates memory with malloc

### Metrics

| Variant | MRR | First relevant rank | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- | --- | --- |
| strict | 0.0000 | n/a | no | no | no | no |
| soft | 0.5000 | 2 | no | yes | yes | yes |

### Ranked Neighbors

| Rank | Score | Function | Purpose | Strict hit | Soft hit |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.0000 | __default_allocate | allocates memory using malloc | no | no |
| 2 | 0.8835 | __default_zero_allocate | allocates zero-initialized memory | no | yes |
| 3 | 0.8835 | __default_zero_allocate | allocates zero-initialized memory | no | yes |
| 4 | 0.8544 | __default_reallocate | reallocates memory block | no | no |
| 5 | 0.8544 | __default_reallocate | reallocates memory using standard realloc | no | no |
| 6 | 0.8445 | rcutils_get_default_allocator | provides default memory allocator | no | no |
| 7 | 0.8445 | rcutils_get_default_allocator | returns default memory allocator | no | no |
| 8 | 0.8083 | __default_deallocate | frees allocated memory | no | no |
| 9 | 0.8083 | __default_deallocate | deallocates memory using standard free | no | no |
| 10 | 0.7683 | rcutils_get_zero_initialized_allocator | provides a zero-initialized allocator | no | no |

---

## Clustering Evaluation

Cluster count: 5
Total items: 420

### Metrics

Weighted purity: 0.0286
Macro purity: 0.0286

### Cluster Sanity Checks

| Check | Passed | Details |
| --- | --- | --- |
| no_empty_clusters | yes | {"empty_clusters": []} |
| no_overly_dominant_cluster | yes | {"largest_cluster_share": 0.23809523809523808, "threshold": 0.5} |
| no_excess_singletons | yes | {"singleton_clusters": 0} |

### Cluster Sizes

Min: 66
Max: 100
Mean: 84.0000
Stddev: 14.8593
Largest cluster share: 0.2381

### Per-Cluster Purity

| Cluster | Size | Dominant label | Purity | Top labels |
| --- | --- | --- | --- | --- |
| 0 | 100 | checks if key exists in string map | 0.0400 | checks if key exists in string map (4), finds last occurrence of delimiter in string (3), advances directory iterator to next entry (2) |
| 1 | 66 | returns zero initialized hash map | 0.0303 | checks if key exists in hash map (2), computes sha 256 sigma1 function (2), finalizes sha 256 hash computation (2) |
| 2 | 94 | allocates zero initialized memory | 0.0213 | allocates zero initialized memory (2), checks if shared library is loaded (2), creates directory at absolute path (2) |
| 3 | 94 | appends severity name to log output | 0.0213 | adds or updates logger severity level in hash map (2), appends function name to logging output (2), appends log message to output buffer (2) |
| 4 | 66 | decodes base64 string to byte array | 0.0303 | decodes base64 string to byte array (2), resizes uint8 array buffer (2), adds element to array list (1) |
