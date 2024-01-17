import numpy as np


def inner_product_matrix(x, y):
    """

    Args:
        x (): array with coordinates of shape [...,n,L] where n = no. of atoms and L no. of discretization points
        y (): array with coordinates of shape [...,n,L] where n = no. of atoms and L no. of discretization points

    Returns: distance vector matrix between atomic position vectors in array of shape [...,n,n]

    """
    n = x.shape[-2]
    return ((np.repeat(x, repeats=n, axis=-2) * np.tile(y, reps=(n, 1))).reshape((y.shape[:-2] + (n, n, -1)))).sum(-1)


def coordinates_to_distance_vectors(x):
    """

    Args:
        x (): array with coordinates of shape [...,n,d] where n = no. of atoms and d = dimension atoms live in

    Returns: distance vector matrix between atomic position vectors in array of shape [...,n,n,d]

    """
    n = x.shape[-2]
    return (np.repeat(x, repeats=n, axis=-2) - np.tile(x, reps=(n, 1))).reshape((x.shape[:-2] + (n, n, -1)))


def coordinates_to_distance_matrix(x):
    """

    Args:
        x (): array with coordinates of shape [...,n,d] where n = no. of atoms and d = dimension atoms live in

    Returns: distance matrix between atomic position vectors in array of shape [...,n,n,1]

    """

    D = np.sum(coordinates_to_distance_vectors(x)**2, axis=-1, keepdims=True)
    return safe_mask(D > 0., np.sqrt, D)


def safe_mask(mask, fn, operand, placeholder=0):
    masked = np.where(mask, operand, 0)
    return np.where(mask, fn(masked), placeholder)
