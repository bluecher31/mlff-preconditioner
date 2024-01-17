import numpy as np
from typing import Tuple, List
from tqdm import trange


def _preprocess_schur_complement(S: np.ndarray, permute) -> Tuple[np.ndarray, int]:
    permutation = 0
    if permute is True:
        S, permutation = _permute_matrix(S)
    return S, permutation


def _permute_matrix(k: np.ndarray) -> Tuple[np.ndarray, int]:
    argmax = -np.argmax(np.diag(k))
    k = np.roll(k, shift=(argmax, argmax), axis=(0, 1))
    return k, -argmax


def _one_schur_update(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if K[0, 0] <= 0:
        raise ValueError(f'negative pivot element: {K[0, 0]}')
    l0 = np.sqrt(K[0, 0])

    l = np.hstack([l0,        # scalar
                  1/l0 * K[1:, 0]])    # n_dimension - 1

    S = K[1:, 1:].copy()
    S -= l[1:, np.newaxis].dot(l[np.newaxis, 1:])
    return l, S


def cholesky_decompostition(k: np.ndarray, permute=False, break_percentage=1.) \
        -> Tuple[np.ndarray, List[int]]:
    """returns (S14) hence the cholesky decomposition in the pivoted version/space."""
    S = np.squeeze(k).copy()
    n_dimension = S.shape[0]
    list_of_permutations = []

    L = np.zeros([n_dimension, n_dimension])
    for i in trange(n_dimension):
        S, permutation = _preprocess_schur_complement(S,  permute=permute)
        l, S = _one_schur_update(S)
        L[i:] = np.roll(L[i:], shift=-permutation, axis=0)
        L[i:, i] = l
        list_of_permutations.append(permutation)

        if break_percentage < (i+1)/n_dimension:
            L[i+1:, i+1:] = np.eye(n_dimension-i-1) * 1. * L[i, i]
            break

    return L, list_of_permutations


def pivot_transformation(k_input: np.ndarray, list_of_permutation: List[int],
                                 forward=False, is_matrix=True) -> np.ndarray:
    """Invert permutation on matrix K starting with the last permutation.
        forward = True: original K to pivoted version.
        forward = False: for pivoted K_hat back to original K
    """
    k = k_input.copy()
    len_list = len(list_of_permutation)
    size_matrix = k.shape[0]
    index_offset = size_matrix - len_list
    assert index_offset >= 0
    if forward is False:
        for i, perm in enumerate(list_of_permutation[::-1]):
            ind = -(index_offset + i + 1)
            k[ind:] = np.roll(k[ind:], shift=perm, axis=0)          # permute rows
            if is_matrix is True:
                k[:, ind:] = np.roll(k[:, ind:], shift=perm, axis=1)    # permute columns

    elif forward is True:
        for ind, perm in enumerate(list_of_permutation):
            k[ind:] = np.roll(k[ind:], shift=-perm, axis=0)          # permute rows
            if is_matrix is True:
                k[:, ind:] = np.roll(k[:, ind:], shift=-perm, axis=1)    # permute columns
    return k


# # check correctness of cholesky with a full random matrix
n = 6
B = np.random.randn(n, n)
A = B.dot(B.T)
L, _ = cholesky_decompostition(A)
if not np.allclose(A, L.dot(L.T)):
    raise ValueError('There is an error in the cholesky_decomposition()')

n = 6
B = np.random.randn(n, n)
A = B.dot(B.T)
L, permutations = cholesky_decompostition(A, permute=True)
k = L.dot(L.T)
k_reversed = pivot_transformation(k, permutations)
if not np.allclose(k_reversed, A):
    raise ValueError('Pivoting does not work')
