import src.data.problems
from src.data import random
from src.data.problems import cifar10, with_nans
from src.model.simpleoption import new_option_model
from src.training.core import train, adam_optimizer
from src.type.core import SimpleTrainingSpec

TRAINING_SPEC = SimpleTrainingSpec(
    validation_holdout_fraction=1 / 4,
    epoch_count=10,
    batch_size=64,
    model_maker=new_option_model,
    optimizer=adam_optimizer(),
)


def main():
    train(with_nans(cifar10()), TRAINING_SPEC)


def train_with_random_data():
    random_dataset_spec = random.DatasetSpec(
        size=250,
        feature_count=2,
        class_count=20,
        nan_proportion=0.2,
    )

    train(src.data.problems.random(random_dataset_spec), TRAINING_SPEC)


if __name__ == "__main__":
    main()
