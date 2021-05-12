from jax.experimental.stax import BatchNorm, Conv, Dense, Flatten, Relu, LogSoftmax
from jax.experimental import stax
import jax.numpy as jnp


def cross_entropy_loss(apply_func):
    def loss(params, batch):
        x, y = batch
        preds = apply_func(params, x)
        return -jnp.mean(jnp.sum(preds * y, axis=1))

    return loss


def MLPModel():
    return stax.serial(
        Dense(2048),
        Relu,
        Dense(1024),
        Relu,
        Dense(1024),
        Relu,
        Dense(10),
        LogSoftmax,
    )


def CNNModel():
    return stax.serial(
        Conv(32, (5, 5), (2, 2), padding="SAME"),
        BatchNorm(),
        Relu,
        Conv(10, (3, 3), (2, 2), padding="SAME"),
        BatchNorm(),
        Relu,
        Conv(10, (3, 3), (2, 2), padding="SAME"),
        Relu,
        Flatten,
        Dense(10),
        LogSoftmax,
    )
