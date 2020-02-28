from dataclasses import replace

from src.commons.imports import AUTOTUNE, tf
from src.commons.python.core import on_first
from src.commons.tensorflow.extention import with_noise
from src.commons.tensorflow.maker import NAN
from src.data.core import from_numpy, normalized
from src.data.random import DatasetSpec, random_dataset
from src.type.core import LearningProblem, IOType


def cifar10() -> LearningProblem:
    train, _ = tf.keras.datasets.cifar10.load_data()
    xs, ys = train
    return LearningProblem.with_default_crossentropy(
        dataset=from_numpy((normalized(xs), ys)),
        io_type=IOType([32, 32, 3], [10])
    )


def random(spec: DatasetSpec) -> LearningProblem:
    return LearningProblem.with_default_crossentropy(
        dataset=random_dataset(spec),
        io_type=IOType([spec.feature_count], [spec.class_count])
    )


def with_nans(problem: LearningProblem, nan_proportion: float = 0.5) -> LearningProblem:
    dtype = tf.dtypes.float64  # TODO
    mapping = on_first(with_noise(tf.cast(NAN, dtype), nan_proportion))
    return replace(
        problem,
        dataset=replace(
            problem.dataset,
            data=problem.data().map(mapping, num_parallel_calls=AUTOTUNE)
        )
    )

# https://patrykchrabaszcz.github.io/Imagenet32/
