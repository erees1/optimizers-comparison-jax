#  from optimizers import SGD
#  import tensorflow_datasets as tfds
from argparse import ArgumentParser
import logging
import os
from pathlib import Path
import shutil
import time

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy.random as npr

import datasets
from model import FullyConnectedNetwork
from optimizers import SGD, SGDWithMomentum, RMSProp

# Jax preallocates 90% of gpu memory by default
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#  from jax import grad, jit


hidden_sizes = [2048, 2048, 1024, 1024, 10]
param_scale = 0.1
lr = 0.001
NUM_EPOCHS = 20
BATCH_SIZE = 128
N_TARGETS = 10


# Logging
sh = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")


def setup_logger(name, log_file=None, level=logging.INFO, sh=None):
    """To setup as many loggers as you want"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # File logger
    if log_file is not None:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Streaming logger
    if sh is not None:
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def accuracy(model, batch):
    """Calculate accuracy of model"""
    (images, targets) = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(model(images), axis=1)
    return jnp.mean(predicted_class == target_class)


def data_stream(images, labels, seed=0):
    """Return batches of data"""
    num_train = images.shape[0]
    num_complete_batches, leftover = divmod(num_train, BATCH_SIZE)
    num_batches = num_complete_batches + bool(leftover)
    rng = npr.RandomState(seed)
    perm = rng.permutation(num_train)
    for i in range(num_batches):
        batch_idx = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        yield images[batch_idx], labels[batch_idx]


def train(n_epochs, optimizer, save_dir=None, log_interval=10, val_interval=10):
    """Train model using specified optimizer"""
    train_images, train_labels, test_images, test_labels = datasets.cifar()
    model = FullyConnectedNetwork(
        [train_images.shape[-1]] + hidden_sizes, "relu", random.PRNGKey(0)
    )

    if optimizer == "sgd_with_momentum":
        optimizer = SGDWithMomentum(model.loss, lr=lr)
    elif optimizer == "sgd":
        optimizer = SGD(model.loss, lr=lr)
    elif optimizer == "rmsprop":
        optimizer = RMSProp(model.loss, lr=lr)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not found")

    step = 0

    # Logging
    sh = logging.StreamHandler()
    loss_logger = setup_logger(
        "loss_logger", os.path.join(save_dir, "loss.log"), sh=sh
    )
    acc_logger = setup_logger(
        "acc_logger", os.path.join(save_dir, "acc.log"), sh=None
    )

    for epoch in range(n_epochs):
        ds = data_stream(train_images, train_labels, seed=epoch)
        for _, batch in enumerate(ds):
            model, loss = optimizer.update(model, batch)
            if step % log_interval == 0:
                loss_logger.info(f"Step {step} train_loss {loss:0.3f}")

            if step % val_interval == 0:
                loss = model.get_loss((test_images, test_labels))
                loss_logger.info(f"Step {step} val_loss {loss:0.3f}")

            step += 1

        train_acc = accuracy(model, (train_images, train_labels))
        test_acc = accuracy(model, (test_images, test_labels))
        acc_logger.info(f"Epoch {epoch} train_acc {train_acc} test_acc {test_acc}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    if args.save_dir is not None:
        if Path(args.save_dir).exists():
            shutil.rmtree(args.save_dir)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train(NUM_EPOCHS, optimizer=args.optimizer, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
