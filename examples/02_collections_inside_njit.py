"""Containers inside @njit: an event-driven simulation sketch.

A PriorityQueue schedules events by time and a SparseSet tracks which
entities are active — both constructed and used entirely inside a
jitted function, at native speed.

Run:  python examples/02_collections_inside_njit.py
"""

import numpy as np
from numba import njit

from numba_utils import PriorityQueue, SparseSet, seed


@njit
def simulate(n_entities: int, n_events: int) -> tuple[int, int]:
    events = PriorityQueue(n_events)
    active = SparseSet(n_entities)

    for _ in range(n_events):
        events.push(np.random.random() * 100.0)

    processed = 0
    while not events.is_empty():
        t = events.pop_min()
        entity = int(t * n_entities / 100.0) % n_entities
        if active.contains(entity):
            active.discard(entity)
        else:
            active.add(entity)
        processed += 1
    return processed, active.size()


def main() -> None:
    seed(42)  # Numba's nopython RNG is separate from NumPy's
    processed, still_active = simulate(1000, 100_000)
    print(f"processed {processed} events, {still_active} entities active")


if __name__ == "__main__":
    main()
