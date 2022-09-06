import jax
import jax.numpy as jnp
import numpy as np

from functools import reduce

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class TensorTrain:
    """Class TensorTrain is a core data structure in implementation of
    TensorTrain in JAX. It is a base class for any specific specific algebraic
    type (like TT-matrix or TT-tensor) which leverage TensorTrain
    representation underneath.

    :param cores: List of TT-cores.
    :param shape: Tensor shape.
    :param rank: TT-rank.

    We maintain the following invariants in this class.

     1. All dtypes of cores are the same.
     2. Number of dimensions, number of cores, number of elements in shape are
        the same. Number of elements in ranks is greater by one.
     3. There is a degenerate case of trivial (empty) shape which corresponds
        to scalar.
    """

    def __init__(self, cores: list[jnp.array], shape, rank):
        if len(cores) != len(shape) and len(cores) != len(rank) - 1:
            raise ValueError('Number of dimensions is inconsistent.')

        dtype = cores[0].dtype
        for i, core in enumerate(cores[1:]):
            if dtype != core.dtype:
                raise ValueError('Cores\' dtype is inconsistent: '
                                 f'dtype of core #0 ({dtype}) does not '
                                 f'match dtype of core #{i}.')

        self.shape = shape
        self.rank = rank
        self.cores = cores

    def __repr__(self) -> str:
        params = f'ndim={self.ndim}, shape={self.shape}, rank={self.rank}'
        return f'{self.__class__.__name__}({params})'

    def __eq__(lhs, rhs) -> bool:
        if lhs.shape != rhs.shape or lhs.rank != rhs.rank:
            return False
        for lc, rc in zip(lhs.cores, rhs.cores):
            if jnp.all(lc != rc):
                return False
        return True

    def __getitem__(self, ix):
        acc = self.cores[0][0, ix[0]]
        for i, core in zip(ix[1:], self.cores[1:]):
            acc = acc @ core[i]
        return acc.squeeze()

    @property
    def dtype(self):
        return jnp.cores[0].dtype

    @property
    def ndim(self):
        return len(self.cores)

    @property
    def size(self):
        return reduce(lambda x, y: x + np.prod(y.shape), self.cores, 0)

    def tolist(self):
        return [core.tolist() for core in self.cores]

    def tree_flatten(self):
        return self.cores, {'shape': self.shape, 'rank': self.rank}

    @classmethod
    def tree_unflatten(cls, treedef, leaves):
        return cls(leaves, **treedef)


def randn(prng: jnp.ndarray, shape: tuple[int],
          rank: tuple[int]) -> TensorTrain:
    if len(shape) < 2:
        raise ValueError('Shape should have at least two dimensions.')
    if rank[0] != 1 or rank[-1] != 1:
        raise ValueError('TT rank should have form (1, ..., 1).')
    cores = []
    for it, dim in enumerate(shape):
        core_shape = (rank[it], dim, rank[it + 1])
        core = jax.random.uniform(prng, core_shape)
        cores.append(core)
    return TensorTrain(cores, shape, rank)
