import numpy as np

SAMPLE_SIZE = 10000


# TODO


def expectation_maximization(dataset, mask):
    mean, cov = initial_gaussian(dataset, mask)

    # Expectation Maximization of Gaussian model
    while True:
        for index, row in enumerate(dataset):
            unknown_mask = mask[index]
            known_mask = np.logical_not(unknown_mask)
            cond_mean, cond_covariance = get_conditional(mean, cov, unknown_mask, row[known_mask])
            dataset[index, unknown_mask] = cond_mean

        new_mean = np.mean(dataset, axis=0)
        new_covariance = np.cov(dataset, rowvar=False)

        # Stop once mean and covariance converged.
        if np.allclose(mean, new_mean) and np.allclose(cov, new_covariance):
            break

        mean = new_mean
        cov = new_covariance

    return mean, cov


def initial_gaussian(dataset, mask):
    dataset_without_incomplete_rows = dataset[np.all(mask, axis=-1)]

    # First guess for mean and covariance, assuming the data is Gaussian
    mean = np.mean(dataset_without_incomplete_rows, axis=0)  # TODO also use values from partial rows?
    cov = np.cov(dataset_without_incomplete_rows, rowvar=False)  # TODO fix in case not enough full rows
    return mean, cov


def get_conditional(mean, cov, unknown_columns_mask, known_values):
    """Returns the mean and covariance of the conditional Gaussian,
    conditioned on the passed known_column_values."""
    m1, m2, c11, c12, c21, c22 = split(mean, cov, unknown_columns_mask)

    intermediate = np.matmul(c12, np.linalg.pinv(c22))
    conditional_mean = m1 + np.matmul(intermediate, known_values - m2)
    conditional_covariance = c11 - np.matmul(intermediate, c21)
    return conditional_mean, conditional_covariance


def split(mean, cov, unknown_columns_mask):
    """Splits the dataset mean and covariance according to the given mask."""
    known_columns_mask = np.logical_not(unknown_columns_mask)

    m1 = mean[unknown_columns_mask]
    m2 = mean[known_columns_mask]

    c11 = cov[unknown_columns_mask][:, unknown_columns_mask]
    c12 = cov[unknown_columns_mask][:, known_columns_mask]
    c21 = cov[known_columns_mask][:, unknown_columns_mask]
    c22 = cov[known_columns_mask][:, known_columns_mask]

    return m1, m2, c11, c12, c21, c22


def probability_of_match(mean, cov, dataset, mask, lower_bounds, upper_bounds, row_index):
    """Returns the probability that the row specified by the given index satisfies the given constraints."""
    unknown_columns_mask = mask[row_index]
    known_columns_mask = np.logical_not(unknown_columns_mask)

    known_values = dataset[row_index][known_columns_mask]
    if not is_in_bounds(lower_bounds[known_columns_mask], upper_bounds[known_columns_mask], known_values):
        return 0  # Probability of mach is zero if a known attribute is out of bounds.
    if np.all(known_columns_mask):
        return 1  # Probability is one if all attributes are known and within bounds.

    # Keep only the relevant bounds.
    lower_bounds = lower_bounds[unknown_columns_mask]
    upper_bounds = upper_bounds[unknown_columns_mask]

    conditional_mean, conditional_cov = get_conditional(mean, cov, unknown_columns_mask, known_values)

    # Do naive Monte Carlo evaluation of probability.
    s = 0
    for sample_point in np.random.multivariate_normal(conditional_mean, conditional_cov, SAMPLE_SIZE):
        if is_in_bounds(lower_bounds, upper_bounds, sample_point):
            s += 1

    return s / SAMPLE_SIZE


def is_in_bounds(lower_bounds, upper_bounds, point):
    """Returns whether the given point lies within the given bounds."""
    return np.all(np.less_equal(lower_bounds, point)) and np.all(np.less_equal(point, upper_bounds))
