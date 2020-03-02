from typing import Tuple

import numpy as np
import statsmodels
from numpy import ma, ndarray
from numpy.ma import MaskedArray

from src.type.numpy import Gaussian

ND_ARRAY_4 = Tuple[ndarray, ndarray, ndarray, ndarray]


# TODO improve statistics (use unbiased estimators, use student distributions, use sampling, ...)
# TODO speed up (parallelize)?
# TODO do same for tensorflow?


def expectation_maximization(dataset: MaskedArray) -> Tuple[MaskedArray, Gaussian]:
    gaussian = initial_gaussian(dataset)
    completed_dataset = imputed(dataset, gaussian)
    while True:
        new_mean = np.mean(completed_dataset, axis=0)
        new_cov = np.cov(completed_dataset, rowvar=False)
        if np.allclose(gaussian.mean, new_mean) and np.allclose(gaussian.cov, new_cov):
            break
        gaussian = Gaussian(new_mean, new_cov)
        completed_dataset = imputed(dataset, gaussian)
    return ma.array(completed_dataset, mask=dataset.mask), gaussian


def initial_gaussian(dataset: MaskedArray) -> Gaussian:
    mean = ma.mean(dataset, axis=0)
    # noinspection PyUnresolvedReferences
    cov = statsmodels.stats.correlation_tools.cov_nearest(
        ma.cov(dataset, rowvar=False),
        method='nearest',
        threshold=1e-15,
        n_fact=128,
        return_all=False
    )
    return Gaussian(mean, cov)


def imputed(dataset: MaskedArray, gaussian: Gaussian) -> ndarray:
    """Completes dataset using maximum likelihood."""
    completed_dataset = dataset.data.copy

    for index, row in enumerate(dataset):
        completed_dataset[index, ma.getmaskarray(row)] = conditional(gaussian, row).mean

    return completed_dataset


def conditional(gaussian: Gaussian, partial_point: MaskedArray) -> Gaussian:
    """Returns the mean and covariance of the conditional Gaussian,
    conditioned on the passed known_column_values."""
    unknown_mask = ma.getmaskarray(partial_point)
    known_mask = ~unknown_mask

    m1, m2 = split_mean(gaussian.mean, unknown_mask, known_mask)
    c11, c12, c21, c22 = split_cov(gaussian.cov, unknown_mask, known_mask)

    intermediate = np.matmul(c12, np.linalg.pinv(c22))
    conditional_mean = m1 + np.matmul(intermediate, partial_point[known_mask] - m2)
    conditional_covariance = c11 - np.matmul(intermediate, c21)
    return Gaussian(conditional_mean, conditional_covariance)


def split_mean(mean: ndarray, unknown_mask: ndarray, known_mask: ndarray) -> Tuple[ndarray, ndarray]:
    """Splits the dataset mean according to the given mask."""
    return mean[unknown_mask], mean[known_mask]


def split_cov(cov: ndarray, unknown_mask: ndarray, known_mask: ndarray) -> ND_ARRAY_4:
    """Splits the dataset covariance according to the given mask."""
    c11 = cov[unknown_mask][:, unknown_mask]
    c12 = cov[unknown_mask][:, known_mask]
    c21 = cov[known_mask][:, unknown_mask]
    c22 = cov[known_mask][:, known_mask]

    return c11, c12, c21, c22
