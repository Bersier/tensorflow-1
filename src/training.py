import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Optimizer

from src.data.utils import split_dataset, ready_for_training, BATCH_SIZE, ready_for_evaluation
from src.imports import tf
from src.model import new_flat_model
from src.split.binarysplit import UnitSplit
from src.types.classes import LearningProblem, SizedDataset

HOLDOUT_FRACTION_FOR_VALIDATION = UnitSplit.from_second(1 / 4)

EPOCH_COUNT = 1


def train(problem: LearningProblem):
    model = model_ready_for_training(problem)

    training_dataset, validation_dataset = split_dataset(
        binary_split=HOLDOUT_FRACTION_FOR_VALIDATION,
        dataset=problem.dataset
    )

    training_dataset = ready_for_training(training_dataset, BATCH_SIZE)
    validation_dataset = ready_for_evaluation(validation_dataset, BATCH_SIZE)

    fit(model, training_dataset, validation_dataset)


def model_ready_for_training(problem: LearningProblem) -> tf.keras.Model:
    optimizer: Optimizer = tfa.optimizers.AdamW(amsgrad=True, weight_decay=1e-6)

    model = new_flat_model(problem.io_type)
    model.compile(
        optimizer=optimizer,
        loss=problem.loss_function,
        metrics=problem.metrics
    )
    model.summary()
    return model


def fit(model: tf.keras.Model, dataset: SizedDataset, validation_dataset: SizedDataset):
    result = model.evaluate(
        validation_dataset.data,
        verbose=0
    )
    print("initial_val_loss: {:6.4f} - initial_val_sparse_categorical_accuracy: {:6.4f}"
          .format(result[0], result[1]))

    model.fit(
        dataset.data,
        verbose=1,
        epochs=EPOCH_COUNT,
        validation_data=validation_dataset.data
    )
