# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Datasets used in examples."""
import pickle
import tarfile
import array
import gzip
import os
from os import path
import struct
import urllib.request

import numpy as np


_DATA = "/tmp/jax_example_data/"


def _download(url, filename):
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print("downloaded {} to {}".format(url, _DATA))


def _partial_flatten(x):
    """Flatten all but the first dimension of an ndarray."""
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = mnist_raw()

    train_images, train_labels = _process(train_images, train_labels)
    test_images, test_labels = _process(test_images, test_labels)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels


def _process(images, labels):
    images = _partial_flatten(images) / np.float32(255.0)
    labels = _one_hot(labels, 10)
    return images, labels


def cifar_raw():
    base_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    def parse(filename):
        with open(filename, "rb") as fh:
            dict = pickle.load(fh, encoding="bytes")
        return dict[b"data"], dict[b"labels"]

    file_name = "cifar-10-python.tar.gz"
    extracted_dir = "cifar-10-batches-py"

    _download(base_url, file_name)
    t = tarfile.open(path.join(_DATA, file_name))
    t.extractall(path.join(_DATA))

    # batches are 1-5 and test batch
    filenames = ["data_batch_" + str(i) for i in range(1, 6)]
    filenames += ["test_batch"]

    X, Y = [], []
    for filename in filenames[:-1]:
        x, y = parse(path.join(_DATA, extracted_dir, filename))
        X.append(x)
        Y.append(y)

    train_images = np.concatenate(X)
    train_labels = np.concatenate(Y)

    test_images, test_labels = parse(path.join(_DATA, extracted_dir, filenames[-1]))
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels


def cifar(permute_train=False):
    train_images, train_labels, test_images, test_labels = cifar_raw()
    train_images, train_labels = _process(train_images, train_labels)
    test_images, test_labels = _process(test_images, test_labels)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels
