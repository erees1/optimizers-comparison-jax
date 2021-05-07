from jax import jit
from jax import value_and_grad
from functools import partial
import jax.numpy as jnp


@jit
def clip_grads(a, abs_val):
    return jnp.clip(a, a_min=-abs_val, a_max=abs_val)


class Optimizer:
    def __init__(self, loss_func, lr=0.001, grad_clip=1):
        self.lr = lr
        self.loss_func = loss_func
        self.grad_clip = grad_clip

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
            p - self.lr * clip_grads(dp, self.grad_clip)
            for p, dp in zip(params, grads)
        ]
        return out, l


class SGDWithMomentum(Optimizer):
    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = None
        self.beta = beta

    def update(self, model, batch):
        if self.momentum is None:
            self.momentum = [jnp.zeros(p.shape) for p in model.params]
        model.params, loss, self.momentum = self._update(
            model.params, batch, self.momentum
        )
        return model, loss

    @partial(jit, static_argnums=(0,))
    def _update(self, params, batch, momentum):
        l, grads = value_and_grad(self.loss_func)(params, batch)

        new_params = []
        new_momentum = []
        for p, dp, mp in zip(params, grads, momentum):
            mp = self.beta * mp - self.lr * clip_grads(dp, self.grad_clip)
            new_params.append(p + mp)
            new_momentum.append(mp)
        return new_params, l, new_momentum


class RMSProp(Optimizer):
    def __init__(self, *args, beta=0.9, epsilon=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.s = None
        self.beta = beta
        self.epsilon = epsilon

    def update(self, model, batch):
        if self.s is None:
            self.s = [jnp.zeros(p.shape) for p in model.params]
        model.params, loss, self.s = self._update(model.params, batch, self.s)
        return model, loss

    @partial(jit, static_argnums=(0,))
    def _update(self, params, batch, s):
        l, grads = value_and_grad(self.loss_func)(params, batch)

        new_params = []
        new_s = []
        for p, dp, sp in zip(params, grads, s):
            dp = clip_grads(dp, self.grad_clip)
            sp = self.beta * sp + (1 - self.beta) * dp * dp
            p = p - self.lr * dp / jnp.sqrt(sp + self.epsilon)
            new_params.append(p)
            new_s.append(sp)

        return new_params, l, new_s


class Adam(Optimizer):
    pass
