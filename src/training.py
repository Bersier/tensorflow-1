from src.imports import tf

EPOCH_COUNT = 10


def train(model: tf.keras.Model, dataset, val_dataset):
    model.fit(
        dataset,
        verbose=1,
        epochs=EPOCH_COUNT,
        validation_data=val_dataset
    )
