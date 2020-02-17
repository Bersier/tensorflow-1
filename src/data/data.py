from src.data.utils import get_dataset_from_numpy
from src.imports import tf


# https://patrykchrabaszcz.github.io/Imagenet32/


def get_cifar10_dataset():
    train, _ = tf.keras.datasets.cifar10.load_data()
    return get_dataset_from_numpy(train)
