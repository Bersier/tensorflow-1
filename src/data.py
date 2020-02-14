import numpy as np

from src.imports import tf


def get_datasets():
    data = np.random.random((1000, 32))
    labels = random_one_hot_labels((1000, 10))

    val_data = np.random.random((100, 32))
    val_labels = random_one_hot_labels((100, 10))

    dataset = tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(1000)
    dataset = dataset.batch(32).repeat()

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(32).repeat()
    return dataset, val_dataset


def random_one_hot_labels(shape):
    n, n_class = shape
    classes = np.random.randint(0, n_class, n)
    labels = np.zeros((n, n_class))
    labels[np.arange(n), classes] = 1
    return labels
