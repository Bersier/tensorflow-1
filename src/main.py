from src.data import learningproblems
from src.training import train


def main():
    train(learningproblems.cifar10())


if __name__ == "__main__":
    main()
