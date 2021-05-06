from jax import jit
from jax import value_and_grad
from functools import partial
import jax.numpy as jnp


class Optimizer:
    def __init__(self, loss_func, lr=0.001):
        self.lr = lr
        self.loss_func = loss_func

    def update(self, model, batch):
        model.params, loss = self._update(model.params, batch)
        return model, loss

    def _update(self, *args):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @partial(jit, static_argnums=(0,))
    def _update(self, params, batch):
        l, grads = value_and_grad(self.loss_func)(params, batch)
        out = [
            (w - self.lr * dw, b - self.lr * db)
            for (w, b), (dw, db) in zip(params, grads)
        ]
        return out, l


class SGDWithMomentum(Optimizer):
    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = None
        self.beta = beta

    def update(self, model, batch):
        sizes = model.sizes
        if self.momentum is None:
            self.momentum = [
                (
                    jnp.zeros((m, n)),
                    jnp.zeros((n,)),
                )
                for m, n in zip(sizes[:-1], sizes[1:])
            ]
        model.params, loss, self.momentum = self._update(
            model.params, batch, self.momentum
        )
        return model, loss

    @partial(jit, static_argnums=(0,))
    def _update(self, params, batch, momentum):
        l, grads = value_and_grad(self.loss_func)(params, batch)

        new_params = []
        new_momentum = []
        for (w, b), (dw, db), (mw, mb) in zip(params, grads, momentum):
            mw = self.beta * mw - self.lr * jnp.clip(dw, a_min=-1, a_max=1)
            mb = self.beta * mb - self.lr * jnp.clip(db, a_min=-1, a_max=1)
            new_params.append((w + mw, b + mb))
            new_momentum.append((mw, mb))

        return new_params, l, new_momentum


class RMSProp(Optimizer):
    pass


class Adam(Optimizer):
    pass
