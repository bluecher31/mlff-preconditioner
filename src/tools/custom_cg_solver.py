import numpy as np
import scipy
from tqdm import trange, tqdm
from typing import Tuple, List


from sgdml.solvers.incomplete_cholesky import pivoted_cholesky, get_col


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

    S = K[1:, 1:]
    S -= l[1:, np.newaxis] @ l[np.newaxis, 1:]
    return l, S


def cholesky_decomposition(k: np.ndarray, permute=False, break_percentage=1.) \
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
        L[i:, i] = l.copy()
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
L, _ = cholesky_decomposition(A)
if not np.allclose(A, L.dot(L.T)):
    raise ValueError('There is an error in the cholesky_decomposition()')

n = 6
B = np.random.randn(n, n)
A = B.dot(B.T)
L, permutations = cholesky_decomposition(A, permute=True)
k = L.dot(L.T)
k_reversed = pivot_transformation(k, permutations)
if not np.allclose(k_reversed, A):
    raise ValueError('Pivoting does not work')


def init_percond_operator(K, lam, break_percentage):
    print(f'\nstart cholesky decomposition: break after {break_percentage * 100:.1f}%')

    max_rank = int(break_percentage * K.shape[0])
    get_col_K = lambda i: get_col(K.copy(), i)
    L, _ = pivoted_cholesky(get_col=get_col_K, diagonal=np.diag(K), max_rank=max_rank)


    kernel = lam * np.eye(max_rank) + (L.T @ L)
    # L2, _ = cholesky_decomposition(kernel, permute=False)  # k x k
    L2 = scipy.linalg.cholesky(kernel, lower=True)           # k x k
    temp_mat = scipy.linalg.solve_triangular(L2, L.T, lower=True)  # k x N

    # del L, K, L_pi

    def apply_inv(a):  # a.shape: N or N x N
        temp_vec = temp_mat.T @ (temp_mat @ a)
        x = lam ** -1 * (a - temp_vec)
        return x

    M = scipy.sparse.linalg.LinearOperator(shape=K.shape, matvec=apply_inv)
    return M


def solve_linear_system_woodbury(K, y, lam, break_percentage=0.1):

    size = K.size * K.itemsize * 2**-20
    print(f'\nstart solve_linear_system_woodbury\n'
          f'K.shape = {K.shape},    memory size: {size:.1f}MB or {size * 2**-10:.1f}GB \n')



    M = init_percond_operator(K, lam, break_percentage)
    K_hat = K + lam * np.eye(K.shape[0])

    global num_iters
    num_iters = 0
    global pbar
    print('\nstart solving conjugate gradient')

    def _cg_status(_: np.ndarray):
        global num_iters
        num_iters += 1
        pbar.update(1)

    pbar = tqdm(total=K.shape[0]*10, desc='CG')
    alphas, info = scipy.sparse.linalg.cg(K_hat, y, M=M, callback=_cg_status)
    if info is not 0:
        raise ValueError(f'CG not converged. Please increase preconditioning')

    pbar.close()
    print(f'\n'
          f'num_iters = {num_iters}\n'
          f'K.shape = {K.shape}\n'
          f'finished CG\n')

    return alphas, num_iters
