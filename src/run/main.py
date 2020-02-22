from src.data import learningproblems
from src.data import random
from src.training.core import train


def main():
    train(learningproblems.random(random.DatasetSpec(
        size=250,
        feature_count=2,
        class_count=20,
        nan_proportion=0.2
    )))


if __name__ == "__main__":
    main()
