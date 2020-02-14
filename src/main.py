import tensorflow_addons as tfa

from src.data import get_datasets
from src.imports import tf
from src.model import new_model
from src.training import train


# import matplotlib.pyplot as plt


def main():
    dataset, val_dataset = get_datasets()

    optimizer = tfa.optimizers.AdamW(amsgrad=True, weight_decay=1e-6)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model = new_model(optimizer, loss)
    model.summary()

    train(model, dataset, val_dataset)


if __name__ == "__main__":
    main()
