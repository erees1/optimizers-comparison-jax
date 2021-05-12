from argparse import ArgumentParser
import logging
import os
from pathlib import Path
import shutil

import jax.numpy as jnp
import jax.random as random
import numpy.random as npr

import datasets
from modelf import CNNModel, MLPModel, cross_entropy_loss
from optimizers import Adam, RMSProp, RMSPropWithMomentum, SGD, SGDWithMomentum

# Jax preallocates 90% of gpu memory by default
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#  from jax import grad, jit


lr = 0.001
BATCH_SIZE = 128
N_TARGETS = 10
SEED = 42
ES_PATIENCE = 1500  # Number of steps to see no improvement in best val loss

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


def accuracy(apply_fun, params, batch):
    """Calculate accuracy of model"""
    (images, targets) = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(apply_fun(params, images), axis=1)
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


def train(
    n_epochs,
    optimizer,
    model="fcn",
    ds="mnist",
    save_dir=None,
    log_interval=10,
    val_interval=10,
):
    """Train model using specified optimizer"""
    flatten = False if model == "cnn" else True

    # Dataset options
    if ds == "cifar":
        train_images, train_labels, test_images, test_labels = datasets.cifar(
            flatten=flatten
        )
    elif ds == "mnist":
        train_images, train_labels, test_images, test_labels = datasets.mnist(
            flatten=flatten
        )
    else:
        raise NotImplementedError("Dataset {ds} not found")

    # Model options
    if model == "fcn":
        init_fun, apply_fun = MLPModel()
    elif model == "cnn":
        init_fun, apply_fun = CNNModel()
    else:
        raise NotImplementedError(f"Model {model} not found")

    _, params = init_fun(
        random.PRNGKey(SEED), (BATCH_SIZE,) + train_images.shape[1:]
    )

    # Optimizer options
    loss_fun = cross_entropy_loss(apply_fun)
    if optimizer == "sgd_momentum":
        optimizer = SGDWithMomentum(loss_fun, lr=lr)
    elif optimizer == "sgd":
        optimizer = SGD(loss_fun, lr=lr)
    elif optimizer == "rmsprop":
        optimizer = RMSProp(loss_fun, lr=lr)
    elif optimizer == "rmsprop_momentum":
        optimizer = RMSPropWithMomentum(loss_fun, lr=lr)
    elif optimizer == "adam":
        optimizer = Adam(loss_fun, lr=lr)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not found")

    step = 0
    best_val_loss = float("inf")
    es_counter = 0
    early_stop = False

    # Logging
    sh = logging.StreamHandler()

    def log_fp(save_dir, name):
        return os.path.join(save_dir, name) if save_dir is not None else None

    loss_logger = setup_logger("loss_logger", log_fp(save_dir, "loss.log"), sh=sh)
    acc_logger = setup_logger("acc_logger", log_fp(save_dir, "acc.log"), sh=None)

    for epoch in range(n_epochs):
        ds = data_stream(train_images, train_labels, seed=epoch)
        for _, batch in enumerate(ds):
            loss, params = optimizer.update(params, batch)

            if step % log_interval == 0:
                loss_logger.info(f"Step {step} train_loss {loss}")

            if step % val_interval == 0:
                loss = loss_fun(params, (test_images, test_labels))
                loss_logger.info(f"Step {step} val_loss {loss}")

                if loss > best_val_loss:
                    es_counter += 1
                else:
                    best_val_loss = loss
                    es_counter = 0

                if es_counter > ES_PATIENCE // val_interval:
                    early_stop = True
                    break

            step += 1

        train_acc = accuracy(apply_fun, params, (train_images, train_labels))
        test_acc = accuracy(apply_fun, params, (test_images, test_labels))
        acc_logger.info(f"Epoch {epoch} train_acc {train_acc} test_acc {test_acc}")

        if early_stop:
            loss_logger.info(
                f"Validation loss has not improved for last {ES_PATIENCE} Steps -> early stopping!"
            )
            return


def main():
    parser = ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--ds", type=str, default="mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="fcn")
    args = parser.parse_args()

    if args.save_dir is not None:
        if Path(args.save_dir).exists():
            shutil.rmtree(args.save_dir)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train(
        args.epochs,
        model=args.model,
        optimizer=args.optimizer,
        save_dir=args.save_dir,
        ds=args.ds,
    )


if __name__ == "__main__":
    main()
