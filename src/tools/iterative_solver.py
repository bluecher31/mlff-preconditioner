import numpy as np
import scipy
from tqdm import tqdm
from typing import Tuple

from sgdml.solvers import incomplete_cholesky as ichol


class Iterative():
    def __init__(self, gdml_train, task,  R_desc, R_d_desc, tril_perms_lin, callback=None
):
        self.strength_preconditioning = 0.1
        self.gdml_train = gdml_train
        self.task = task

        self.R_desc = R_desc
        self.R_d_desc = R_d_desc
        self.tril_perms_lin = tril_perms_lin

        pass

    def solve(self, K: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        solve the linear system K*x = y

        Returns
        -------
        alphas, num_iters, train_rmse:
            solution array, number of cg iterations, remaining error
        """
        # get preconditioner
        M = self._init_precon_operator()

        # calculate inversion with CG
        global num_iters
        num_iters = 0
        alphas = self._cg(K, y, M=M)

        # calculate remaining error
        train_rmse = np.linalg.norm(K_hat @ alphas - y)
        return alphas, num_iters, train_rmse

    def _init_precon_operator(self) -> scipy.linalg.LinearOperator:
        lam = self.task['lam']
        lam_inv = 1./lam

        print(f'\nstart cholesky decomposition: break after {self.strength_preconditioning * 100:.1f}%')

        # TODO: directly access get_col and diag for kernel matrix
        k = int(self.strength_preconditioning * K.shape[0])
        get_col_K = lambda i: ichol.get_col(K, i)
        L, _ = ichol.pivoted_cholesky(get_col=get_col_K, diagonal=np.diag(K), max_rank=k)

        print('set-up preconditioning')
        K_hat = K + lam * np.eye(K.shape[0])

        kernel = lam * np.eye(k) + (L.T @ L)
        L2, _ = scipy.linalg.cholesky(kernel, lower=True)  # k x k
        temp_mat = scipy.linalg.solve_triangular(L2, L.T, lower=True)  # k x N

        # del L, K, L_pi

        def apply_inv(a):  # a.shape: N or N x N
            temp_vec = temp_mat.T @ (temp_mat @ a)
            x = lam_inv * (a - temp_vec)
            return x

        M = scipy.sparse.linalg.LinearOperator(shape=K_hat.shape, matvec=apply_inv)
        return M

    def _cg(self):
        global pbar

        def _cg_status(_: np.ndarray):
            global num_iters
            num_iters += 1
            pbar.update(1)

        pbar = tqdm(total=K_hat.shape[0] * 10, desc='Solve CG')
        # TODO: insert linear operator instead of K_hat
        # K_hat = K + lam*eye
        alphas, info = scipy.sparse.linalg.cg(K_hat, y, M=M, callback=_cg_status)
        if info != 0:
            raise ValueError(f'CG not converged. Please increase preconditioning')
        pbar.close()

        print(f'\n'
              f'num_iters = {num_iters}\n'
              f'K.shape = {K_hat.shape}\n'
              f'finished CG\n')
        return alphas

    def _get_col_kernel_mat(self, i_col: int) -> np.ndarray:

        k = self.gdml_train._assemble_kernel_mat(
            self.R_desc,
            self.R_d_desc,
            self.tril_perms_lin,
            self.task['sig'],
            # self.desc,
            # use_E_cstr=use_E_cstr,
            col_idxs=[i_col],
            # callback=callback,
        )
        return k

