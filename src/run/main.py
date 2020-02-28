import src.data.problems
from src.data import random
from src.data.problems import cifar10, with_nans
from src.training.core import train


def main():
    train(with_nans(cifar10()))


def train_with_random_data():
    train(src.data.problems.random(random.DatasetSpec(
        size=250,
        feature_count=2,
        class_count=20,
        nan_proportion=0.2
    )))


if __name__ == "__main__":
    main()
