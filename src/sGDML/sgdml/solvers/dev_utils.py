import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator

from tqdm import trange

def get_eigvals(K_op: LinearOperator, P_op: LinearOperator) -> np.ndarray:
    print('calculate eigvals')
    assert K_op.shape == P_op.shape
    n = K_op.shape[0]
    vec = np.zeros(n)
    # P_K = np.zeros((n, n))
    # for i in range(n):
    #     vec[i] = 1
    #     P_K[i] = P_op @ (K_op @ vec)
    #     vec[i] = 0

    K = np.zeros((n, n))
    for i in range(n):
        vec[i] = 1
        K[i] = -K_op @ vec
        vec[i] = 0
    P_K = P_op @ K
    eigvals_P = scipy.linalg.eigvals(P_K)

    # import matplotlib.pyplot as plt
    # eigvals_K = scipy.linalg.eigvals(K)
    # plt.figure()
    # plt.plot(np.abs(eigvals_K)/np.abs(eigvals_K).min(), label='K')
    # plt.plot(np.abs(eigvals_P), label='P')
    # plt.legend()
    # plt.semilogy()
    # U, s, Vh = scipy.linalg.svd(P_K)
    # Uk, sk, Vhk = scipy.linalg.svd(K)
    #
    # n_datapoints = 25
    # i_col = 76
    # factor = 1
    #
    # randomness = []
    # for i_col in range(n_datapoints*63):
    #     uk0 = np.zeros((factor*63, n_datapoints))
    #     for i in range(n_datapoints):
    #         uk0[:, i] = Uk[i*63:(i+factor)*63, i_col]
    #     measure = np.sum(uk0.std(axis=1))
    #     randomness.append(measure)
    # plt_mat(uk0)
    # plt.title(f'i_col = {i_col}')
    #
    # def plt_mat(mat):
    #     plt.figure()
    #     vmax = mat.max()
    #     plt.imshow(mat, cmap="seismic", vmax=vmax, vmin=-vmax)
    #     plt.colorbar()
    #     plt.tight_layout()

    return eigvals_P

