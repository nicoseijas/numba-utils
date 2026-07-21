import heapq

import numpy as np
import pytest
from numba import complex128, float32, float64, int64, njit

from numba_utils.collections import (
    BitSet,
    FixedQueue,
    ObjectPool,
    PriorityQueue,
    RingBuffer,
    SparseSet,
    Stack,
    counter,
    fixed_queue_type,
    priority_queue_type,
    ring_buffer_type,
    stack_type,
    typed_defaultdict,
)


class TestTypedDefaultdict:
    def test_usable_from_python_and_njit(self):
        d = typed_defaultdict(key_type=int64, value_type=float64)
        d[3] = 1.5

        @njit
        def bump(dd, key):
            dd[key] = dd.get(key, 0.0) + 1.0

        bump(d, 3)
        bump(d, 7)
        assert d[3] == 2.5
        assert d[7] == 1.0

    def test_rejects_non_numba_types(self):
        with pytest.raises(TypeError):
            typed_defaultdict(int, float)


class TestCounter:
    def test_matches_reference_counts(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 20, 5000)
        counts = counter(arr)
        expected = np.bincount(arr, minlength=20)
        for value in range(20):
            assert counts.get(value, 0) == expected[value]

    def test_float_keys(self):
        counts = counter(np.array([1.5, 1.5, 2.0]))
        assert counts[1.5] == 2
        assert counts[2.0] == 1


class TestBitSet:
    def test_matches_python_set(self):
        rng = np.random.default_rng(1)
        bits = BitSet(500)
        reference = set()
        for value in rng.integers(0, 500, 2000):
            v = int(value)
            if v in reference:
                bits.discard(v)
                reference.discard(v)
            else:
                bits.add(v)
                reference.add(v)
        assert bits.count() == len(reference)
        for v in range(500):
            assert bits.contains(v) == (v in reference)

    def test_clear(self):
        bits = BitSet(100)
        bits.add(3)
        bits.add(64)
        bits.clear()
        assert bits.count() == 0

    def test_errors(self):
        with pytest.raises(ValueError):
            BitSet(0)
        bits = BitSet(10)
        with pytest.raises(IndexError):
            bits.add(10)
        with pytest.raises(IndexError):
            bits.contains(-1)


class TestStack:
    def test_lifo_order(self):
        stack = Stack(3)
        stack.push(1.0)
        stack.push(2.0)
        assert stack.peek() == 2.0
        assert stack.pop() == 2.0
        assert stack.pop() == 1.0
        assert stack.is_empty()

    def test_overflow_and_underflow(self):
        stack = Stack(1)
        stack.push(1.0)
        with pytest.raises(ValueError):
            stack.push(2.0)
        stack.pop()
        with pytest.raises(ValueError):
            stack.pop()

    def test_usable_inside_njit(self):
        @njit
        def sum_via_stack(values):
            stack = Stack(values.shape[0])
            for v in values:
                stack.push(v)
            total = 0.0
            while not stack.is_empty():
                total += stack.pop()
            return total

        assert sum_via_stack(np.array([1.0, 2.0, 3.0])) == 6.0


class TestFixedQueue:
    def test_fifo_order(self):
        queue = FixedQueue(3)
        queue.push(1.0)
        queue.push(2.0)
        assert queue.pop() == 1.0
        queue.push(3.0)
        queue.push(4.0)
        assert queue.pop() == 2.0
        assert queue.pop() == 3.0
        assert queue.pop() == 4.0

    def test_wraparound_stress(self):
        queue = FixedQueue(5)
        expected = []
        for i in range(100):
            if queue.is_full() or (i % 3 == 0 and not queue.is_empty()):
                assert queue.pop() == expected.pop(0)
            else:
                queue.push(float(i))
                expected.append(float(i))
        assert queue.size() == len(expected)

    def test_overflow_and_underflow(self):
        queue = FixedQueue(1)
        with pytest.raises(ValueError):
            queue.pop()
        queue.push(1.0)
        with pytest.raises(ValueError):
            queue.push(2.0)


class TestRingBuffer:
    def test_overwrites_oldest(self):
        ring = RingBuffer(3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ring.push(v)
        assert ring.size() == 3
        np.testing.assert_array_equal(ring.to_array(), [3.0, 4.0, 5.0])

    def test_last_indexing(self):
        ring = RingBuffer(3)
        for v in [1.0, 2.0, 3.0, 4.0]:
            ring.push(v)
        assert ring.last(0) == 4.0
        assert ring.last(2) == 2.0
        with pytest.raises(IndexError):
            ring.last(3)

    def test_partial_fill(self):
        ring = RingBuffer(10)
        ring.push(1.0)
        ring.push(2.0)
        np.testing.assert_array_equal(ring.to_array(), [1.0, 2.0])
        assert not ring.is_full()


class TestPriorityQueue:
    def test_matches_heapq(self):
        rng = np.random.default_rng(2)
        values = rng.normal(0, 1, 500)
        pq = PriorityQueue(500)
        reference = []
        for v in values:
            pq.push(v)
            heapq.heappush(reference, v)
        popped = [pq.pop_min() for _ in range(500)]
        expected = [heapq.heappop(reference) for _ in range(500)]
        np.testing.assert_array_equal(popped, expected)

    def test_peek_does_not_remove(self):
        pq = PriorityQueue(5)
        pq.push(2.0)
        pq.push(1.0)
        assert pq.peek_min() == 1.0
        assert pq.size() == 2

    def test_overflow_and_underflow(self):
        pq = PriorityQueue(1)
        with pytest.raises(ValueError):
            pq.pop_min()
        pq.push(1.0)
        with pytest.raises(ValueError):
            pq.push(2.0)

    def test_usable_inside_njit(self):
        @njit
        def smallest_two(values):
            pq = PriorityQueue(values.shape[0])
            for v in values:
                pq.push(v)
            return pq.pop_min(), pq.pop_min()

        a, b = smallest_two(np.array([5.0, 1.0, 3.0]))
        assert (a, b) == (1.0, 3.0)


class TestSparseSet:
    def test_add_discard_contains(self):
        sset = SparseSet(100)
        sset.add(5)
        sset.add(50)
        sset.add(5)
        assert sset.size() == 2
        assert sset.contains(5) and sset.contains(50)
        sset.discard(5)
        assert not sset.contains(5)
        sset.discard(5)
        assert sset.size() == 1

    def test_matches_python_set_under_churn(self):
        rng = np.random.default_rng(3)
        sset = SparseSet(200)
        reference = set()
        for value in rng.integers(0, 200, 3000):
            v = int(value)
            if v in reference:
                sset.discard(v)
                reference.discard(v)
            else:
                sset.add(v)
                reference.add(v)
        assert sset.size() == len(reference)
        assert set(sset.values().tolist()) == reference

    def test_clear_is_constant_time_reset(self):
        sset = SparseSet(10)
        sset.add(3)
        sset.clear()
        assert sset.size() == 0
        assert not sset.contains(3)

    def test_out_of_range_raises(self):
        sset = SparseSet(10)
        with pytest.raises(IndexError):
            sset.add(10)


class TestObjectPool:
    def test_acquire_release_cycle(self):
        pool = ObjectPool(3)
        slots = [pool.acquire() for _ in range(3)]
        assert sorted(slots) == [0, 1, 2]
        assert pool.n_available() == 0
        pool.release(slots[1])
        assert pool.n_available() == 1
        assert pool.acquire() == slots[1]

    def test_exhaustion_raises(self):
        pool = ObjectPool(1)
        pool.acquire()
        with pytest.raises(ValueError):
            pool.acquire()

    def test_double_release_raises(self):
        pool = ObjectPool(2)
        slot = pool.acquire()
        pool.release(slot)
        with pytest.raises(ValueError):
            pool.release(slot)

    def test_out_of_range_raises(self):
        pool = ObjectPool(2)
        with pytest.raises(IndexError):
            pool.release(5)


class TestDtypeGenericFactories:
    def test_float64_specialization_is_the_default_class(self):
        assert stack_type(float64) is Stack
        assert fixed_queue_type(float64) is FixedQueue
        assert ring_buffer_type(float64) is RingBuffer
        assert priority_queue_type(float64) is PriorityQueue

    def test_specializations_are_cached(self):
        assert stack_type(int64) is stack_type(int64)
        assert priority_queue_type(int64) is priority_queue_type(int64)

    def test_int64_stack_preserves_values(self):
        IntStack = stack_type(int64)
        s = IntStack(3)
        s.push(2**62)
        # float64 storage would round 2**62 + 1; int64 must not.
        s.push(2**62 + 1)
        assert s.pop() == 2**62 + 1
        assert s.pop() == 2**62

    def test_int64_priority_queue_orders_exactly(self):
        IntPQ = priority_queue_type(int64)
        pq = IntPQ(8)
        values = [5, -3, 2**62 + 1, 2**62, 0]
        for v in values:
            pq.push(v)
        drained = [pq.pop_min() for _ in range(len(values))]
        assert drained == sorted(values)

    def test_float32_ring_buffer_dtype(self):
        RB = ring_buffer_type(float32)
        rb = RB(4)
        for v in (1.5, 2.5, 3.5):
            rb.push(v)
        out = rb.to_array()
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, [1.5, 2.5, 3.5])

    def test_fixed_queue_int_specialization_fifo(self):
        IntQueue = fixed_queue_type(int64)
        q = IntQueue(2)
        q.push(10)
        q.push(20)
        assert q.pop() == 10
        q.push(30)
        assert q.pop() == 20
        assert q.pop() == 30

    def test_usable_inside_njit(self):
        IntStack = stack_type(int64)

        @njit
        def roundtrip():
            s = IntStack(4)
            s.push(7)
            s.push(9)
            return s.pop() + s.pop()

        assert roundtrip() == 16

    def test_rejects_non_numba_types(self):
        for factory in (
            stack_type,
            fixed_queue_type,
            ring_buffer_type,
            priority_queue_type,
        ):
            with pytest.raises(TypeError):
                factory(np.int64)
            with pytest.raises(TypeError):
                factory(int)
            # unhashable arguments get the friendly message, not the
            # cache dict's "unhashable type"
            with pytest.raises(TypeError, match="numba scalar type"):
                factory([1, 2])

    def test_priority_queue_rejects_complex(self):
        with pytest.raises(TypeError):
            priority_queue_type(complex128)
        # unordered containers accept complex
        stack_type(complex128)
