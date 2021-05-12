from typing import Any, Tuple, List
from abc import abstractmethod
from jax import jit
from jax import value_and_grad
from functools import partial
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

Array = Any


@jit
def clip_grads(a, abs_val):
    return jnp.clip(a, a_min=-abs_val, a_max=abs_val)


class Optimizer:
    def __init__(self, loss_func, lr=0.001, grad_clip=1):
        self.lr = lr
        self.loss_func = loss_func
        self.grad_clip = grad_clip
        # Once initalized state should be a pytree with same shape as params
        self.state = None
        self.step = 0

    def update(self, params: List[Any], batch: Tuple[Array, Array]):
        self.step += 1
        if self.state is None:
            self.state = self.init_optimizer_state(params)
        l, params, self.state = self._update(params, batch, self.state)
        return l, params

    @partial(jit, static_argnums=(0,))
    def _update(self, params, batch: Tuple[Array, Array], state: List[Any]):
        l, grads = value_and_grad(self.loss_func)(params, batch)
        param_leaves, treedef = tree_flatten(params)
        grad_leaves = treedef.flatten_up_to(grads)
        state_leaves = treedef.flatten_up_to(state)
        all_results = zip(
            *[
                self._clip_and_update_param_array(p, g, state)
                for p, g, state in zip(param_leaves, grad_leaves, state_leaves)
            ]
        )
        params, state = [treedef.unflatten(r) for r in all_results]
        return l, params, state

    @partial(jit, static_argnums=(0,))
    def _clip_and_update_param_array(self, p, g, *args) -> Tuple[Any, Tuple]:
        return self._update_param_array(p, clip_grads(g, self.grad_clip), *args)

    @abstractmethod
    def init_optimizer_state(self, params: List[Any]) -> List[Any]:
        # How the optimizer should create state
        params_flat, tree_def = tree_flatten(params)
        empty_tree = [()] * len(params_flat)
        return tree_unflatten(tree_def, empty_tree)

    @abstractmethod
    def _update_param_array(self, p: Array, g: Array, *args) -> Tuple[Any, Tuple]:
        # Must implement this method for each optimizer
        # Should return tuple of (new_params, optimizer_state)
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_param_array(self, p, g, _) -> Tuple[Any, Tuple]:
        return p - self.lr * g, ()


class SGDWithMomentum(Optimizer):
    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = None
        self.beta = beta

    def init_optimizer_state(self, params):
        # Initialize tree of same shape as params but all zeros
        return tree_map(lambda x: x * 0.0, params)

    def _update_param_array(self, p, g, state) -> Tuple[Any, Tuple[Any]]:
        m = state
        m = self.beta * m - self.lr * g
        p = p + m
        return p, m


class RMSProp(Optimizer):
    def __init__(self, *args, beta=0.9, epsilon=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = None
        self.beta = beta
        self.epsilon = epsilon

    def init_optimizer_state(self, params):
        # Initialize tree of same shape as params but all zeros
        return tree_map(lambda x: x * 0.0, params)

    @partial(jit, static_argnums=(0,))
    def _update_param_array(self, p, g, state) -> Tuple[Any, Tuple[Any]]:
        sq_grads = state
        sq_grads = self.beta * sq_grads + (1 - self.beta) * jnp.square(g)
        p = p - self.lr * g / jnp.sqrt(sq_grads + self.epsilon)
        return p, sq_grads


class RMSPropWithMomentum(Optimizer):
    def __init__(self, *args, beta1=0.9, beta2=0.9, epsilon=1e-6, **kwargs):
        """
        beta1 governs moving average of square gradients and beta2 is the
        momentum parameters
        """
        super().__init__(*args, **kwargs)
        self.s = None
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def init_optimizer_state(self, params):
        # Initialize tree of same shape as params but all zeros
        state = tree_map(lambda x: (x * 0.0, x * 0.0), params)
        return state

    @partial(jit, static_argnums=(0,))
    def _update_param_array(self, p, g, state) -> Tuple[Any, Tuple[Any, ...]]:
        sq_grads, mom = state
        sq_grads = self.beta1 * sq_grads + (1 - self.beta1) * jnp.square(g)
        rmsg = g / jnp.sqrt(sq_grads + self.epsilon)
        mom = self.beta2 * mom - self.lr * rmsg
        p = p + mom
        return p, (sq_grads, mom)

class Adam(Optimizer):
    def __init__(self, *args, beta1=0.9, beta2=0.999, epsilon=1e-6, **kwargs):
        """
        beta1 governs momentum and beta2 governs moving avg of squares
        """
        super().__init__(*args, **kwargs)
        self.s = None
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def init_optimizer_state(self, params):
        # Initialize tree of same shape as params but all zeros
        state = tree_map(lambda x: (x * 0.0, x * 0.0), params)
        return state

    @partial(jit, static_argnums=(0,))
    def _update_param_array(self, p, g, state) -> Tuple[Any, Tuple[Any, ...]]:
        s, m = state
        m = self.beta1 * m - (1 - self.beta1) * g
        s = self.beta2 * s + (1 - self.beta2) * jnp.square(g)
        m_hat = m / (1 - self.beta1 ** (self.step + 1))
        s_hat = s / (1 - self.beta2 ** (self.step + 1))
        p = p + self.lr * m_hat / jnp.sqrt(s_hat + self.epsilon)
        return p, (s, m)
