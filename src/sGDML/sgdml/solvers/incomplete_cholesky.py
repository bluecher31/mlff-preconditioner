import numpy as np
from typing import Tuple, List, Callable
from tqdm import trange
import time
import pickle

measure_time_cholesky = True


def _swap_entries(vec: np.ndarray, i: int, j: int) -> None:
    """swaps i, j'th entry of vec inplace"""
    vec[i], vec[j] = vec[j], vec[i]


def _check_inputs(get_col: Callable[[int], np.ndarray], diagonal: np.ndarray, max_rank: int) -> None:
    """sensity check on inputs"""
    temp = get_col(0)       # test call
    assert temp.ndim == 1, 'get_col returns array with more than one dimension'
    assert diagonal.ndim == 1, 'diag with more than one dimension'
    assert temp.shape[0] == diagonal.size, 'dimension from get_col and diag do not match'
    assert max_rank <= diagonal.size, f'max_rank = {max_rank} is too large'


def pivoted_cholesky(get_col: Callable[[int], np.ndarray], diagonal: np.ndarray, max_rank: int) \
        -> [np.ndarray, np.ndarray, dict]:
    """
    calculates the pivoted cholesky decomposition
    Parameters
    ----------
    get_col: function which returns column i from a PSD matrix K
    diagonal: np.diag(K)
    max_rank: specifies the rank of the approximation

    Returns
    -------
    L: np.ndarray, shape = n x max_rank
    index_columns: perturbations which turn L into a true lower triangular matrix
    """
    _check_inputs(get_col=get_col, diagonal=diagonal, max_rank=max_rank)

    # init variables
    diag = diagonal.copy()          # decouple from outside
    n = diag.size                   # size of column
    index_columns = np.arange(n)    # original index list
    error = np.linalg.norm(diag, ord=1)
    L = np.zeros((n, max_rank))

    log_time = np.zeros(max_rank)

    for m in trange(max_rank, desc='pivoted_cholesky'):
        tic_start = time.perf_counter()
        # find new pivot element in diagonal
        i_argmax = int(np.argmax(diag[index_columns][m:]) + m)
        # permute order in index labeling
        _swap_entries(index_columns, m, i_argmax)

        m_pi = index_columns[m]         # index for pivot element
        i_pi = index_columns[m+1:]      # index array for remaining column entries

        # save new pivot element
        pivot_element = diag[m_pi]
        assert pivot_element > 0, f'given matrix is not PSD, remaining error = {error:.2e}'
        L[m_pi, m] = np.sqrt(pivot_element)

        # get column from original matrix
        k = get_col(m_pi)

        # calculate correction from previous cholesky factors
        schur_factor = 0
        if m > 0:
            # schur_factor = np.sum(L[m_pi, :m] * L[i_pi, :m], axis=1)
            schur_factor = np.einsum('c, rc->r', L[m_pi, :m], L[i_pi, :m])

        # calculate new cholesky factor
        L[i_pi, m] = (k[i_pi] - schur_factor)/L[m_pi, m]

        # update diagonal and remaining error
        diag[i_pi] -= L[i_pi, m] ** 2
        toc_end = time.perf_counter()
        log_time[m] = toc_end - tic_start       # in seconds

    error = np.linalg.norm(diag[m+1:], ord=1)
    print(f'Remaining error on diagonal = {error:.2e}')
    size = L.size * L.itemsize * 2**-20
    print(f'L.shape = {L.shape}, memory size: {size:.1f}MB or {size* 2**-10:.1f}GB')
    info_cholesky = {'time_cholesky': log_time,
                     'L.shape': L.shape,
                     'index_columns': index_columns}

    # if measure_time_cholesky is True:
    #     print('saved log_time')
    #     pickle.dump(info_cholesky, open('time_cholesky', "wb"))
    return L, index_columns, info_cholesky


def get_col(mat: np.ndarray, i_col: int) -> np.ndarray:
    """
    returns a single column of mat
    Parameters
    ----------
    mat: a two-dimensional matrix
    i_col: specifies which col to return

    Returns
    -------
    vec: specified column
    """
    assert mat.ndim == 2
    return mat[:, i_col]
