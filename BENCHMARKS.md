# Benchmarks

Python 3.13.3, numba 0.66.0, numpy 2.4.6, AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD.

Seed 42, 11 runs after 2 warmup runs, mean times.

| case | NumPy | numba-utils | speedup |
| --- | ---: | ---: | ---: |
| fast_clip (2,000,000 f64) | 2.26 ms | 2.25 ms | 1.01x |
| cumulative_sum (2,000,000 f64) | 5.86 ms | 2.58 ms | 2.27x |
| rolling_mean w=50 (2,000,000 f64) | 12.50 ms | 3.05 ms | 4.10x |
| topk k=100 (2,000,000 f64) | 5.86 ms | 0.72 ms | 8.19x |
| histogram 64 bins (2,000,000 f64) | 13.70 ms | 1.46 ms | 9.37x |
| radix_sort full-range (5,000,000 i64) | 49.98 ms | 60.99 ms | 0.82x |
| radix_sort range<2^24 (5,000,000 i64) | 51.57 ms | 39.12 ms | 1.32x |
| counting_sort range<1000 (5,000,000 i64) | 26.06 ms | 10.39 ms | 2.51x |
| unique_sorted (5,000,000 i64, sorted) | 27.53 ms | 1.63 ms | 16.92x |
