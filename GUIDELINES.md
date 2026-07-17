# Development Guidelines

## 1. Performance > Features

Never add a function without a clear performance or ergonomics justification.

## 2. Benchmarks are mandatory

Every PR must include benchmarks.

Not accepted:

> "it seems faster"

Accepted:

| 10M floats  | time  |
| ----------- | ----- |
| NumPy       | 18 ms |
| numba-utils | 9 ms  |

## 3. Numba compatibility

Everything must work with `@njit`. No hacks.

## 4. No dependencies

Ideally only:

- `numpy`
- `numba`

Nothing else.

## 5. Minimal API

Avoid functions with twenty parameters.

Prefer:

```python
topk(arr, 10)
```

Over:

```python
topk(
    arr,
    k=10,
    ascending=False,
    inplace=True,
    stable=False,
    ...
)
```

### The `out=` convention

Array functions return a new array by default, and accept an optional `out=`
parameter to reuse buffers in hot loops without allocations (the NumPy
pattern). When `out` is passed, the function writes into it and returns it.

## 6. Explicit typing

Never rely on implicit conversions.

## 7. Backward compatibility

Break APIs only in major versions.

## 8. Reproducible benchmarks

Pin:

- seed
- input size
- number of iterations
- hardware (when relevant)

Published results must be replicable.

## 9. Usefulness before novelty

Don't add "interesting" algorithms just because they exist. Every addition must solve a recurring problem for Numba users.

## 10. Quality over quantity

Better to offer 80 excellent, well-documented, thoroughly tested functions than 300 shallow utilities.

## Documentation

Every function must follow exactly the same structure:

1. **What**
2. **Why**
3. **Complexity**
4. **Memory**
5. **Example**
6. **Benchmark**
7. **Limitations**
8. **Related functions**

### Example

`fast_argpartition()`

> Finds the k smallest elements without fully sorting the array.
>
> **Complexity:** Average O(n), Worst O(n²)
>
> **Memory:** O(1)
