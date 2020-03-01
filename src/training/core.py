import tensorflow_addons as tfa

from src.commons.imports import tf
from src.data.core import ready_for_training, ready_for_evaluation, split_dataset
from src.split.binarysplit import UnitSplit
from src.type.core import LearningProblem, SizedDataset, SimpleTrainingSpec


def train(problem: LearningProblem, training_spec: SimpleTrainingSpec):
    model = model_ready_for_training(problem, training_spec)

    training_dataset, validation_dataset = split_dataset(
        binary_split=UnitSplit.from_second(training_spec.validation_holdout_fraction),
        dataset=problem.dataset
    )

    training_dataset = ready_for_training(training_dataset, training_spec.batch_size)
    validation_dataset = ready_for_evaluation(validation_dataset, training_spec.batch_size)

    fit(model, training_dataset, validation_dataset, training_spec.epoch_count)


def model_ready_for_training(problem: LearningProblem, training_spec: SimpleTrainingSpec) -> tf.keras.Model:
    model = training_spec.model_maker(problem.io_type)
    model.compile(
        optimizer=training_spec.optimizer,
        loss=problem.loss_function,
        metrics=problem.metrics
    )
    model.summary()
    return model


def fit(model: tf.keras.Model, dataset: SizedDataset, validation_dataset: SizedDataset, epoch_count: int):
    result = model.evaluate(
        validation_dataset.data,
        verbose=0
    )
    print("initial_val_loss: {:6.4f} - initial_val_sparse_categorical_accuracy: {:6.4f}"
          .format(result[0], result[1]))

    model.fit(
        dataset.data,
        verbose=1,
        epochs=epoch_count,
        validation_data=validation_dataset.data
    )


def adam_optimizer(learning_rate=1e-3, weight_decay=1e-6, beta_1=0.9):
    return tfa.optimizers.AdamW(
        amsgrad=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta_1=beta_1
    )
