"""In this test suit we verify implementation of __getitem__ method (or its
free function analogue) as well as benchmark it in several scenarios. The first
scenario is getting a single random element in TT-tensor. The second scenario
is getting a batch of random elements in TT-tensor. Each scenario is run with
or without JIT if applicable (not applicable in case of TTPy).
"""

import jax
import jax.numpy as jnp
import numba
import numpy as np
import pytest
import teneva
import tt
import ttax

from babax import randn


class TestGetItemJAX:
    """Test suit for benchmarking item getter for naive TT-tensor
    implementation of in JAX.
    """

    nosamples = 128
    nodims = 8

    @pytest.fixture
    def tensor(self):
        prng = jax.random.PRNGKey(42)
        rank = (1, ) + (2, ) * (self.nodims - 1) + (1, )
        shape = (2, ) * self.nodims
        tensor = randn(prng, shape, rank)
        return tensor

    @pytest.fixture
    def indexes(self):
        indexes = np.random.randint(0, 2, size=(self.nosamples, self.nodims))
        return jnp.array(indexes)

    @pytest.mark.skip
    @pytest.mark.parametrize('use_jit', [False, True])
    def test_baseline(self, benchmark, indexes, tensor, use_jit):
        current = 0

        def fn():
            nonlocal current
            ix = indexes[current, :]
            el = ix
            current = (current + 1) % self.nosamples
            return el

        # Warm up and benchmark jitted function.
        if use_jit:
            fn = jax.jit(fn)
            fn()
        benchmark(fn)

    @pytest.mark.single
    @pytest.mark.parametrize('use_jit', [False, True])
    def test_getitem(self, benchmark, indexes, tensor, use_jit):
        current = 0
        getter = tensor.__getitem__
        getter = jax.jit(tensor.__getitem__)

        def fn():
            nonlocal current
            ix = indexes[current, :]
            el = getter(ix)
            current = (current + 1) % self.nosamples
            return el

        # Warm up jitted function if required.
        if use_jit:
            fn = jax.jit(fn)
            fn()
        else:
            getter(indexes[0, :])

        # Start benchmarking.
        benchmark(fn)

    @pytest.mark.batched
    @pytest.mark.parametrize('use_jit', [False, True])
    def test_getitems(self, benchmark, indexes, tensor, use_jit):
        getter = jax.jit(tensor.__getitem__)

        def fn():
            acc = 0
            for it in range(indexes.shape[0]):
                ix = indexes[it, :]
                acc += getter(ix)
            return acc

        # Use jit if it required and warm up compiled function.
        if use_jit:
            fn = jax.jit(fn)
            fn()

        # Start benchmarking.
        benchmark(fn)


class TestGetItemTTAX:
    """Test suit for benchmarking item getter of ttax package.
    """

    nosamples = 128
    nodims = 8

    @pytest.fixture
    def indexes(self):
        indexes = np.random.randint(0, 2, size=(self.nosamples, self.nodims))
        return jnp.array(indexes)

    @pytest.fixture
    def tensor(self):
        return teneva.tensor_rand([2] * self.nodims, 2)

    @pytest.mark.skip(reason='Bug in ttax (see '
                      'https://github.com/fasghq/ttax/issues/40 for details).')
    @pytest.mark.single
    @pytest.mark.parametrize('use_jit', [False, True])
    def test_getitem(self, benchmark, indexes, tensor, use_jit):
        current = 0
        getter = jax.jit(tensor.__getitem__)

        def fn() -> float:
            nonlocal current
            ix = indexes[current, :]
            el = getter(ix)
            current = (current + 1) % nosamples
            return el

        # Warm up jitted function.
        if use_jit:
            fn = jax.jit(fn)
            fn()
        else:
            getter()

        # Start benchmarking.
        benchmark(fn)


class TestGetItemTeneva:
    """Test suit for benchmarking item getter of teneva package.
    """

    nosamples = 128
    nodims = 8

    @pytest.fixture
    def tensor(self):
        return teneva.tensor_rand([2] * self.nodims, 2)

    @pytest.fixture
    def indexes(self):
        return np.random.randint(0, 2, size=(self.nosamples, self.nodims))

    @pytest.mark.skip(reason='too fast')
    def test_baseline_jitted(self, benchmark, indexes, tensor):
        current = 0
        nosamples = self.nosamples  # Help numba to infer type of variable.

        @numba.jit
        def fn() -> float:
            nonlocal current
            ix = indexes[current, :]
            el = ix
            current = (current + 1) % nosamples
            return el

        # Warm up and benchmark jitted function.
        fn()
        benchmark(fn)

    @pytest.mark.single
    @pytest.mark.parametrize('use_jit', [False, True])
    def test_getitem(self, benchmark, indexes, tensor, use_jit):
        current = 0
        getter = teneva.getter(tensor)
        nosamples = self.nosamples  # Help numba to infer type of variable.

        def fn() -> float:
            nonlocal current
            ix = indexes[current, :]
            el = getter(ix)
            current = (current + 1) % nosamples
            return el

        # Warm up and benchmark jitted function.
        if use_jit:
            fn = numba.jit(fn)
            fn()
        benchmark(fn)

    @pytest.mark.batched
    @pytest.mark.parametrize('use_jit', [False, True])
    def test_getitems(self, benchmark, indexes, tensor, use_jit):
        getter = teneva.getter(tensor)

        def fn() -> float:
            acc = 0
            for it in range(indexes.shape[0]):
                ix = indexes[it, :]
                acc += getter(ix)
            return acc

        # Warm up and benchmark jitted function.
        if use_jit:
            fn = numba.jit(nopython=True)(fn)
            fn()

        # Start benchmarking.
        benchmark(fn)


class TestGetItemTTPy:
    """Test suit for benchmarking item getter of TT-tensor from ttpy package.
    """

    nosamples = 128
    nodims = 8

    @pytest.fixture
    def tensor(self):
        return tt.rand(n=2, d=self.nodims, r=2)

    @pytest.fixture
    def indexes(self):
        indexes = np.random.randint(0, 2, size=(self.nosamples, self.nodims))
        return indexes

    @pytest.mark.single
    def test_getitem(self, benchmark, indexes, tensor):
        current = 0
        getter = tensor.__getitem__

        @benchmark
        def fn():
            nonlocal current
            ix = indexes[current, :]
            el = getter(ix)
            current = (current + 1) % self.nosamples
            return el

    @pytest.mark.batched
    def test_getitems(self, benchmark, indexes, tensor):
        getter = tensor.__getitem__

        @benchmark
        def fn() -> float:
            acc = 0
            for it in range(indexes.shape[0]):
                ix = indexes[it, :]
                acc += getter(ix)
            return acc
