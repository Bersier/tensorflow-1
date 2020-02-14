from src.data import get_datasets
from src.model import new_model
from src.training import train


# import matplotlib.pyplot as plt


def main():
    dataset, val_dataset = get_datasets()

    model = new_model()
    model.summary()

    train(model, dataset, val_dataset)


if __name__ == "__main__":
    main()
