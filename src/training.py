import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Optimizer

from src.data.utils import split_dataset, ready_for_training, BATCH_SIZE, ready_for_evaluation
from src.imports import tf
from src.model import new_flat_model
from src.split.binarysplit import UnitSplit
from src.types.classes import LearningProblem

EPOCH_COUNT = 10


def train(problem: LearningProblem):
    optimizer: Optimizer = tfa.optimizers.AdamW(amsgrad=True, weight_decay=1e-6)

    model = new_flat_model(problem.model_spec())
    model.compile(
        optimizer=optimizer,
        loss=problem.loss_function,
        metrics=problem.metrics
    )
    model.summary()

    holdout_fraction = UnitSplit.from_second(1 / 4)

    training_dataset, validation_dataset = split_dataset(holdout_fraction, problem.dataset)

    training_dataset = ready_for_training(training_dataset, BATCH_SIZE)
    validation_dataset = ready_for_evaluation(validation_dataset, BATCH_SIZE)

    fit(model, training_dataset.data, validation_dataset.data)


def fit(model: tf.keras.Model, dataset, val_dataset):
    # TODO print initial metrics

    model.fit(
        dataset,
        verbose=1,
        epochs=EPOCH_COUNT,
        validation_data=val_dataset
    )

    # TODO print final metrics
