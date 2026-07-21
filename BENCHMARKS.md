# Benchmarks

Python 3.13.3, numba 0.66.0, numpy 2.4.6, AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD.

Seed 42, 11 runs after 2 warmup runs, mean times.

| case | NumPy | numba-utils | speedup |
| --- | ---: | ---: | ---: |
| fast_clip (2,000,000 f64) | 2.22 ms | 2.24 ms | 0.99x |
| cumulative_sum (2,000,000 f64) | 5.82 ms | 2.57 ms | 2.27x |
| rolling_mean w=50 (2,000,000 f64) | 12.65 ms | 3.02 ms | 4.20x |
| topk k=100 (2,000,000 f64) | 5.83 ms | 0.69 ms | 8.45x |
| histogram 64 bins (2,000,000 f64) | 9.71 ms | 1.42 ms | 6.84x |
| radix_sort full-range (5,000,000 i64) | 50.56 ms | 61.24 ms | 0.83x |
| radix_sort range<2^24 (5,000,000 i64) | 52.19 ms | 40.17 ms | 1.30x |
| counting_sort range<1000 (5,000,000 i64) | 27.05 ms | 13.87 ms | 1.95x |
| unique_sorted (5,000,000 i64, sorted) | 27.75 ms | 1.63 ms | 16.99x |
| combination_table C(30,4)=27,405 (vs itertools) | 0.87 ms | 0.14 ms | 6.39x |
| disjoint_rank_aggregate one-shot (N=M=3,000, K=5) (vs dense NumPy) | 199.35 ms | 62.52 ms | 3.19x |
| DisjointRankStructure.eval (build amortized, same size) | 194.52 ms | 1.47 ms | 132.72x |
| stable_argsort (5,000,000 i64, many ties) | 254.99 ms | 1010.74 ms | 0.25x |
| lexsort 3 keys (1,000,000 i64) | 72.82 ms | 190.13 ms | 0.38x |

`stable_argsort` and `lexsort` lose from Python by design and we ship
them anyway: NumPy's `kind="stable"` uses an O(n) radix sort for
integers, while Numba's only stable kind is mergesort. Their value is
availability INSIDE `@njit` — `np.lexsort` does not exist in nopython
code at all, and calling out to Python from a jitted loop costs far
more than the difference above.

## Random & collections

| case | baseline | numba-utils | speedup |
| --- | ---: | ---: | ---: |
| shuffle (1,000,000 f64) (vs NumPy) | 13.11 ms | 8.50 ms | 1.54x |
| weighted_sampling (10,000 w, 100,000 draws) (vs NumPy) | 6.57 ms | 5.91 ms | 1.11x |
| alias_sample (setup amortized, 100,000 draws) (vs NumPy) | 6.48 ms | 3.31 ms | 1.96x |
| counter (1,000,000 i64, 1k distinct) (vs NumPy) | 5.72 ms | 24.25 ms | 0.24x |
| philox_uniforms (1,000,000 f64, fresh stream) (vs NumPy Philox) | 19.62 ms | 4.09 ms | 4.80x |
| sample_without_replacement 7 of 52 (vs Generator.choice, 1001 runs) | 4.6 µs | 0.6 µs | 7.59x |
| PriorityQueue push+pop (50,000) (vs heapq) | 17.02 ms | 4.39 ms | 3.88x |
| SparseSet churn (200,000 ops) (vs Python set) | 15.46 ms | 5.09 ms | 3.04x |

`counter` loses to sort-based `np.unique` on one-shot counting by design: its use case is incremental counting inside jitted loops, where materializing an array first is the expensive part.

## Parallel (24 threads)

| case | serial baseline | parallel | speedup |
| --- | ---: | ---: | ---: |
| parallel_sum (20,000,000 f64) vs np.sum | 14.82 ms | 2.46 ms | 6.02x |
| parallel_histogram 64 bins (20,000,000 f64) vs serial histogram | 18.43 ms | 3.25 ms | 5.68x |
| parallel_prefix_sum (20,000,000 f64) vs serial cumulative_sum | 28.04 ms | 20.42 ms | 1.37x |
| parallel_topk k=100 (20,000,000 f64) vs serial topk | 11.58 ms | 2.50 ms | 4.63x |
| chunked_reduce MC (20,000,000 philox draws, 256 chunks) vs its serial driver | 145.51 ms | 12.41 ms | 11.73x |

Parallel float reductions reorder operations; results can differ from serial in the last bits (parallel_histogram is bit-exact, and chunked_reduce is bit-exact between its own serial and parallel drivers at fixed n_chunks). Gains depend on core count and memory bandwidth. The philox_uniforms baseline is NumPy's own counter-based workflow (a fresh Philox stream per key/counter, as a per-chunk MC worker builds it).

## Graph

| case | baseline | numba-utils | speedup |
| --- | ---: | ---: | ---: |
| bfs (50,000 nodes, 200,000 edges) (vs Python deque) | 11.43 ms | 0.96 ms | 11.89x |
| dijkstra (50,000 nodes, 200,000 edges) (vs heapq) | 74.62 ms | 6.86 ms | 10.88x |
| UnionFind churn (200,000 unions) (vs Python DSU) | 50.62 ms | 2.54 ms | 19.93x |

Baselines are idiomatic pure Python over prebuilt adjacency lists (the
realistic representation on each side); there is no NumPy equivalent
for these. With SciPy available, `scipy.sparse.csgraph` is the
vectorized alternative from Python — the value here is running inside
`@njit` kernels without leaving nopython.

## Stats

| case | baseline | numba-utils | speedup |
| --- | ---: | ---: | ---: |
| logsumexp (2,000,000 f64) (vs naive NumPy) | 7.13 ms | 6.12 ms | 1.17x |
| softmax (2,000,000 f64) (vs max-shifted NumPy) | 12.02 ms | 7.93 ms | 1.52x |
| weighted_quantile (1,000,000 f64) (vs np.quantile weighted) | 30.82 ms | 87.80 ms | 0.35x |

The naive logsumexp baseline is also numerically WRONG beyond ~709
(exp overflows to inf) — these functions exist for correctness first.
`weighted_quantile` loses from Python: NumPy's SIMD sort outruns
Numba's argsort. Its value is availability inside `@njit`
(`np.quantile` with weights cannot be called from nopython) plus
fail-fast validation of NaN and negative weights.
