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
from optimizers import SGD, SGDWithMomentum

# Jax preallocates 90% of gpu memory by default
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#  from jax import grad, jit


hidden_sizes = [2048, 2048, 1024, 1024, 10]
param_scale = 0.1
lr = 0.001
NUM_EPOCHS = 30
BATCH_SIZE = 128
N_TARGETS = 10


logger = logging.getLogger()
formatter = logging.Formatter("%(message)s")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


def add_file_logging(save_dir):
    """Set up file based logging"""
    fh = logging.FileHandler(os.path.join(save_dir, "log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)


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


def train(n_epochs, optimizer, log_interval=10, val_interval=10):
    """Train model using specified optimizer"""
    train_images, train_labels, test_images, test_labels = datasets.cifar()
    model = FullyConnectedNetwork(
        [train_images.shape[-1]] + hidden_sizes, "relu", random.PRNGKey(0)
    )

    if optimizer == "sgd_with_momentum":
        optimizer = SGDWithMomentum(model.loss, lr=lr)
    elif optimizer == "sgd":
        optimizer = SGD(model.loss, lr=lr)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not found")

    step = 0
    # Keep tracking of accuracy
    train_acc_record = []
    test_acc_record = []

    for epoch in range(n_epochs):
        ds = data_stream(train_images, train_labels, seed=epoch)
        start_time = time.time()
        for _, batch in enumerate(ds):
            model, loss = optimizer.update(model, batch)
            if step % log_interval == 0:
                logging.info(f"Step {step} train_loss {loss:0.3f}")

            if step % val_interval == 0:
                loss = model.get_loss((test_images, test_labels))
                logging.info(f"Step {step} val_loss {loss:0.3f}")

            step += 1

        #  epoch_time = time.time() - start_time
        #  train_acc = accuracy(model, (train_images, train_labels))
        #  test_acc = accuracy(model, (test_images, test_labels))
        #  train_acc_record.append(train_acc)
        #  test_acc_record.append(test_acc)
        #  logging.info(
        #      f"Epoch {epoch + 1} time {epoch_time:0.2f}s train_acc {train_acc:0.3f} test_acc {test_acc:0.3f} "
        #  )

    #  _, ax = plt.subplots()
    #  ax.plot(train_acc_record, label="Training accuracy")
    #  ax.plot(test_acc_record, label="Testing accuracy")
    #  ax.legend()
    #  plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    if args.save_dir is not None:
        if Path(args.save_dir).exists():
            shutil.rmtree(args.save_dir)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        add_file_logging(args.save_dir)

    train(NUM_EPOCHS, optimizer=args.optimizer)


if __name__ == "__main__":
    main()
