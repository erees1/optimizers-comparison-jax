from typing import List
from functools import partial
from jax.scipy.special import logsumexp
import jax.random as random
from jax import numpy as jnp
from jax import jit


# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-1):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (m, n)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


@jit
def relu(x):
    return jnp.maximum(0, x)


class FullyConnectedNetwork:
    def __init__(self, sizes: List[int], activations: str, key):
        self.sizes = sizes
        self.activations = activations
        self.params = init_network_params(self.sizes, key)

    def __call__(self, x):
        return self.predict(self.params, x)

    @partial(jit, static_argnums=(0,))
    def predict(self, params, image):
        activations = image
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            activations = relu(outputs)

        final_w, final_b = params[-1]
        logits = jnp.dot(activations, final_w) + final_b
        return logits - logsumexp(logits, axis=1, keepdims=True)

    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch):
        x, y = batch
        preds = self.predict(params, x)
        return -jnp.mean(jnp.sum(preds * y, axis=1))

    def get_loss(self, batch):
        return self.loss(self.params, batch)
