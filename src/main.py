import tensorflow_addons as tfa

from src.data.random import random_dataset
from src.imports import tf
from src.model import new_model
from src.training import train


def main():
    dataset, val_dataset = random_dataset(2666)

    optimizer = tfa.optimizers.AdamW(amsgrad=True, weight_decay=1e-6)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model = new_model(optimizer, loss)
    model.summary()

    train(model, dataset.data, val_dataset.data)


if __name__ == "__main__":
    main()
