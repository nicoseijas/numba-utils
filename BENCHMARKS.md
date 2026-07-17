# Benchmarks

Python 3.13.3, numba 0.66.0, numpy 2.4.6, AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD.

Seed 42, 11 runs after 2 warmup runs, mean times.

| case | NumPy | numba-utils | speedup |
| --- | ---: | ---: | ---: |
| fast_clip (2,000,000 f64) | 2.22 ms | 2.21 ms | 1.01x |
| cumulative_sum (2,000,000 f64) | 5.83 ms | 2.54 ms | 2.30x |
| rolling_mean w=50 (2,000,000 f64) | 13.34 ms | 3.86 ms | 3.45x |
| topk k=100 (2,000,000 f64) | 5.44 ms | 0.70 ms | 7.82x |
| histogram 64 bins (2,000,000 f64) | 10.66 ms | 1.43 ms | 7.48x |
| radix_sort full-range (5,000,000 i64) | 50.45 ms | 59.55 ms | 0.85x |
| radix_sort range<2^24 (5,000,000 i64) | 51.82 ms | 41.24 ms | 1.26x |
| counting_sort range<1000 (5,000,000 i64) | 25.75 ms | 10.17 ms | 2.53x |
| unique_sorted (5,000,000 i64, sorted) | 28.33 ms | 1.65 ms | 17.21x |

## Random & collections

| case | baseline | numba-utils | speedup |
| --- | ---: | ---: | ---: |
| shuffle (1,000,000 f64) (vs NumPy) | 12.59 ms | 8.46 ms | 1.49x |
| weighted_sampling (10,000 w, 100,000 draws) (vs NumPy) | 6.50 ms | 5.90 ms | 1.10x |
| alias_sample (setup amortized, 100,000 draws) (vs NumPy) | 6.67 ms | 2.54 ms | 2.63x |
| counter (1,000,000 i64, 1k distinct) (vs NumPy) | 5.95 ms | 22.86 ms | 0.26x |
| PriorityQueue push+pop (50,000) (vs heapq) | 21.18 ms | 4.61 ms | 4.60x |
| SparseSet churn (200,000 ops) (vs Python set) | 15.56 ms | 5.10 ms | 3.05x |

`counter` loses to sort-based `np.unique` on one-shot counting by design: its use case is incremental counting inside jitted loops, where materializing an array first is the expensive part.
