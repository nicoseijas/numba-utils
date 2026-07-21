# Weighted sampling in O(1): the alias method

Every Monte Carlo workload eventually hits the same inner loop: sample
an index with probability proportional to a weight vector, millions of
times. Mixed strategies in game solvers, particle filters, categorical
draws in simulations — the shape is identical, and it is almost always
on the hot path.

There is a classic escalation of answers, and the last one — Walker's
alias method — still surprises people: after an O(n) setup, every draw
costs **one uniform, one comparison, and at most one table lookup**,
no matter how many outcomes there are. Constant time, forever.

## The escalation

**Linear scan** — draw `u = uniform(0, total)`, walk the weights until
the running sum passes `u`. O(n) per draw. Fine for one draw; a
disaster for a million.

**Cumulative sums + binary search** — precompute the cumulative
weights once, then each draw is a binary search: O(log n). This is
what `np.random.choice` does internally, and it is the right default:

```python
cum = np.cumsum(weights)
def draw():
    return np.searchsorted(cum, np.random.random() * cum[-1], side="right")
```

**The alias method** — O(n) setup, O(1) per draw. For the "many draws
from fixed weights" workload, this is the endgame.

## The intuition: leveling the histogram

Picture the probability mass as a bar chart with n columns. Some bars
are taller than average, some shorter. Now level it: pour probability
from the tall bars into the short ones until every column has exactly
the same height — with the rule that **each column may contain at most
two outcomes**: its original resident plus one donor (its *alias*).

That construction is always possible (a classic invariant: while any
bar is above average, some other bar is below), and it turns sampling
into two trivially cheap steps:

1. pick a column uniformly — `i = randint(n)`,
2. flip a biased coin — return the resident with probability
   `prob[i]`, otherwise return `alias[i]`.

No search. Two array reads. The dependence on n is gone.

## Building the tables (Vose's version)

The robust construction keeps two worklists — indices whose scaled
weight is below 1 (small) and the rest (large) — and repeatedly pairs
one of each: the small one is finalized with its current mass, and the
large one donates the difference, possibly becoming small itself.

```python
scaled = weights * n / total          # average column becomes 1.0
small, large = [], []                 # partition by scaled < 1.0
while small and large:
    s, g = small.pop(), large.pop()
    prob[s] = scaled[s]               # resident keeps its mass
    alias[s] = g                      # the rest of the column is g's
    scaled[g] -= 1.0 - scaled[s]      # g donated that much
    (small if scaled[g] < 1.0 else large).append(g)
# leftovers (rounding) are exactly full columns
```

Floating-point rounding can leave a few entries marginally on the
wrong list at the end; finalizing leftovers with `prob = 1.0` absorbs
that harmlessly. This is Vose's contribution — the naive construction
can accumulate error; this one degrades gracefully.

## The numbers, honestly

From this project's benchmark suite (10,000 weights, 100,000 draws,
jitted end to end; setup amortized):

| approach | time |
| --- | ---: |
| `np.random.choice` | 6.48 ms |
| cumsum + binary search (jitted) | 5.91 ms |
| alias method (jitted) | 3.31 ms |

Two things worth noticing. Binary search is already close to NumPy —
log(10,000) ≈ 13 comparisons per draw is cheap. The alias method wins
by roughly 2× here, and the gap *widens* with n: its per-draw cost is
flat while binary search grows logarithmically and, more importantly,
scatters its memory accesses across the whole cumulative array. The
alias tables touch exactly two cache lines per draw.

## When NOT to use it

The alias method is a specialist, and the setup cost is real:

- **Weights change between draws** — rebuild is O(n); binary search
  with an updated Fenwick tree, or plain cumsum if updates are rare,
  wins.
- **Few draws** — below n draws or so, the O(n) setup never pays for
  itself. Use binary search.
- **Sampling without replacement** — the alias tables cannot remove an
  outcome; that is a different algorithm (reservoir sampling, or
  sequential draws with weight zeroing).

## The bug class that comes with it

Weighted-sampling code has a signature failure mode worth naming: a
single **NaN weight** slips through the obvious validation. `w < 0` is
`False` for NaN — NaN fails *every* comparison — so a "no negative
weights" check happily accepts it, the running total becomes NaN, and
the whole table degenerates. The symptom is brutal for a Monte Carlo
system: every draw silently returns index 0, and nothing raises. Your
simulation converges confidently to garbage.

Validate weights with `isfinite`, not just sign checks, *before*
building any sampling structure. This exact bug — found by an audit,
not by a failing test — is why the implementations in
[numba-utils](https://github.com/nicoseijas/numba-utils) reject
non-finite weights up front: `alias_setup` builds the tables,
`alias_draw`/`alias_sample` consume them, all callable inside `@njit`.
The setup/draw split is deliberate — a combined "sampler object" would
hide the amortization decision that makes the method worth using.

---

*Related: [Why most Numba benchmarks are wrong](why-most-numba-benchmarks-are-wrong.md) ·
[RNG design notes](../design/rng.md) — including why Numba's nopython
RNG state is separate from NumPy's and must be seeded separately.*
