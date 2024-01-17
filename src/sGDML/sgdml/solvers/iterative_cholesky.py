from functools import partial

import numpy as np
import scipy
import timeit

from .. import DONE, NOT_DONE
from ..predict import GDMLPredict

from sgdml.solvers import incomplete_cholesky as ichol
from tqdm import tqdm, trange
from typing import Tuple

try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True


class CGRestartException(Exception):
    pass


class Iterative(object):
    def __init__(self, gdml_train, desc, task, callback=None, max_processes=None, use_torch=False):

        self.gdml_train = gdml_train
        self.task = task
        self.gdml_predict = None
        self.desc = desc
        self.K_op = None

        n_train, n_atoms = task['R_train'].shape[:2]
        self.n = 3 * n_atoms * n_train     # kernel dimension: n = K.shape[0]
        self._e = np.zeros(self.n)      # vector used to extract columns from self.K_op

        self.callback = callback

        self._max_processes = max_processes
        self._use_torch = use_torch

        # this will be set once the kernel operator is used on the GPU with pytorch
        self._gpu_batch_size = 0

        if self._use_torch and _has_torch:
            print('Using PyTorch in Iterative()')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'device = {device}\n'
                  f'torch.cuda.device_count() = {torch.cuda.device_count()}')

    def solve(
            self,
            task,
            R_desc,
            R_d_desc,
            tril_perms_lin,
            y,
            break_percentage=0.1
    ):
        self.K_op = self._set_up_K_op(task, R_desc, R_d_desc, tril_perms_lin, lam=task['lam'])
        self.n = self.K_op.shape[0]
        diag_K = self._assemble_kernel_mat_diag(tril_perms_lin=tril_perms_lin, sig=task['sig'], R_desc=R_desc,
                                                R_d_desc=R_d_desc, n=self.n)

        self._measure_get_col()

        P_op = self._init_precon_operator(diag_K, self.K_op, lam_regularization=task['lam'],
                                          break_percentage=break_percentage)

        global num_iters
        alphas = self._cg(K_op=self.K_op, y=y, P_op=P_op, tol=task['solver_tol'])     # use internal self.K_op
        return alphas, num_iters

    def _cg(self, K_op, y: np.ndarray, P_op, tol: float) -> Tuple[np.ndarray, int]:
        """
        computes 'alphas = K_op^-1 @ y' using scipy.sparse.linalg.cg and preconditiones with P_op
        Parameters
        ----------
        K_op: LinearOperator or np.ndarray, if None use self.K_op
        y:
        P_op: preconditioner as LinearOperator
        tol: tolerance for solving cg

        Returns
        -------

        """
        global num_iters, pbar

        if K_op is None:
            K_op = self.K_op

        def _cg_status(_: np.ndarray):
            global num_iters
            num_iters += 1
            pbar.update(1)

        num_iters = 0
        pbar = tqdm(total=y.shape[0], desc='CG')
        alphas, info = scipy.sparse.linalg.cg(
            K_op,
            y,
            M=P_op,
            tol=tol,
            callback=_cg_status,
        )
        pbar.close()

        assert info == 0, 'cg did not converge'
        alphas = -alphas
        return alphas

    def _init_precon_operator(self, diag_K, K_op, lam_regularization, break_percentage=0.1):
        """
        computes a preconditioner using an incomplete cholesky decomposition
        Parameters
        ----------
        task
        R_desc
        R_d_desc
        tril_perms_lin
        break_percentage: define accuracy of preconditioner

        Returns
        -------
        M: LinearOperator
        """
        assert len(diag_K) == self.n, 'incorrect dimensions'
        self.K_op = K_op

        lam_inv = 1. / lam_regularization
        # n = self._get_col_K(0).shape[0]  # matrix size
        k = int(break_percentage * self.n)  # max_rank

        print(f'\nstart cholesky decomposition: break after {break_percentage * 100:.1f}%')
        L, _, info_cholesky = ichol.pivoted_cholesky(get_col=self._get_col_K, diagonal=diag_K, max_rank=k)

        # precompute inner matrices
        kernel = lam_regularization * np.eye(k) + (L.T @ L)
        L2 = scipy.linalg.cholesky(kernel, lower=True)  # k x k
        temp_mat = scipy.linalg.solve_triangular(L2, L.T, lower=True)  # k x N

        def apply_inv(a):  # a.shape: N or N x N
            temp_vec = temp_mat.T @ (temp_mat @ a)
            x = lam_inv * (a - temp_vec)
            return x

        return scipy.sparse.linalg.LinearOperator(shape=self.K_op.shape, matvec=apply_inv), info_cholesky

    def _get_col_K(self, i: int):
        self._e[i] = 1
        column = self.K_op.matvec(self._e)
        self._e[i] = 0
        return column

    def _measure_get_col(self, n_repeat=25):
        def random_col():
            i = int(np.random.random_integers(0, 200, 1))
            _ = self._get_col_K(i)

        result = timeit.timeit("random_col()", globals=locals(), number=n_repeat)/n_repeat
        print(f'average (n_repeat = {n_repeat}) time for get_col = {result:.4f}s')

    def _set_up_K_op(self, task, R_desc, R_d_desc, tril_perms_lin, lam):
        """
        returns a scipy.linalg.LinearOperator which resembles K
        lam: regularization added to diagonal of K
        """
        def callback(*args, **kwargs): pass

        n = int(R_d_desc.size / 10)  # matrix size
        K_op = self._init_kernel_operator(
            task, R_desc, R_d_desc, tril_perms_lin, lam, n, callback=callback
        )
        return -K_op

    def _init_kernel_operator(
        self, task, R_desc, R_d_desc, tril_perms_lin, lam, n, callback=None
    ):
        n_train = R_desc.shape[0]

        # dummy alphas
        v_F = np.zeros((n, 1))
        v_E = np.zeros((n_train, 1)) if task['use_E_cstr'] else None

        # Note: The standard deviation is set to 1.0, because we are predicting normalized labels here.
        model = self.gdml_train.create_model(
            task, 'cg', R_desc, R_d_desc, tril_perms_lin, 1.0, v_F, alphas_E=v_E
        )

        self.gdml_predict = GDMLPredict(
            model, max_processes=self._max_processes, use_torch=self._use_torch
        )

        if not self._use_torch:

            if callback is not None:
                callback = partial(callback, disp_str='Optimizing CPU parallelization')
                callback(NOT_DONE)

            self.gdml_predict.prepare_parallel(n_bulk=n_train)

            if callback is not None:
                callback(DONE)

        global is_primed
        is_primed = False

        def _K_vec(v):

            global is_primed
            if not is_primed:
                is_primed = True
                return v

            v_F, v_E = v, None
            if task['use_E_cstr']:
                v_F, v_E = v[:-n_train], v[-n_train:]

            self.gdml_predict.set_alphas(R_d_desc, v_F, alphas_E=v_E)

            if self._use_torch:
                self._gpu_batch_size = self.gdml_predict.get_GPU_batch()

                if self._gpu_batch_size > n_train:
                    self._gpu_batch_size = n_train

            R = task['R_train'].reshape(n_train, -1)
            e_pred, f_pred = self.gdml_predict.predict(R, R_desc, R_d_desc)

            pred = f_pred.ravel()
            if task['use_E_cstr']:
                pred = np.hstack((pred, -e_pred))

            return pred - lam * v

        return scipy.sparse.linalg.LinearOperator((n, n), matvec=_K_vec)

    def _assemble_kernel_mat_diag(self, tril_perms_lin, sig, R_desc, R_d_desc, n, use_E_cstr=False, cols_m_limit=None):
        r"""
        Compute diagonal of the force field kernel matrix.
        Adapted from _assemble_kernel_mat_wkr in train.py

        Parameters
        ----------
            tril_perms_lin : :obj:`numpy.ndarray`
                1D array (int) containing all recovered permutations
                expanded as one large permutation to be applied to a tiled
                copy of the object to be permuted.
            sig : int
                Hyper-parameter :math:`\sigma`.
            use_E_cstr : bool, optional
                True: include energy constraints in the kernel,
                False: default (s)GDML kernel.
            cols_m_limit : int, optional
                Limit the number of columns (include training points 1-`M`).
                Note that each training points consists of multiple columns.

        Returns
        -------
            diag(K)
        """
        # K = np.zeros(K_shape)
        diag_K = np.zeros(n)
        desc_func = self.desc

        n_train, dim_d = R_d_desc.shape[:2]
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        n_perms = int(len(tril_perms_lin) / dim_d)

        for j in trange(n_train, desc='diag(K)'):
            if type(j) is tuple:  # selective/"fancy" indexing

                (
                    K_j,
                    j,
                    keep_idxs_3n,
                ) = j  # (block index in final K, block index global, indices of partials within block)
                blk_j = slice(K_j, K_j + len(keep_idxs_3n))

            else:  # sequential indexing
                blk_j = slice(j * dim_i, (j + 1) * dim_i)
                keep_idxs_3n = slice(None)  # same as [:]

            # TODO: document this exception
            if use_E_cstr and not (cols_m_limit is None or cols_m_limit == n_train):
                raise ValueError(
                    '\'use_E_cstr\'- and \'cols_m_limit\'-parameters are mutually exclusive!'
                )

            # Create permutated variants of 'rj_desc' and 'rj_d_desc'.
            rj_desc_perms = np.reshape(
                np.tile(R_desc[j, :], n_perms)[tril_perms_lin], (n_perms, -1), order='F'
            )

            rj_d_desc = desc_func.d_desc_from_comp(R_d_desc[j, :, :])[0][
                        :, keep_idxs_3n
                        ]  # convert descriptor back to full representation
            rj_d_desc_perms = np.reshape(
                np.tile(rj_d_desc.T, n_perms)[:, tril_perms_lin], (-1, dim_d, n_perms)
            )

            mat52_base_div = 3 * sig ** 4
            sqrt5 = np.sqrt(5.0)
            sig_pow2 = sig ** 2

            # === for i in range(0, n_train):   ======
            # originally 'i' was looped here
            i = j
            # blk_i = slice(i * dim_i, (i + 1) * dim_i)

            diff_ab_perms = R_desc[i, :] - rj_desc_perms
            norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

            mat52_base_perms = np.exp(-norm_ab_perms / sig) / mat52_base_div * 5

            diff_ab_outer_perms = 5 * np.einsum(
                'ki,kj->ij',
                diff_ab_perms * mat52_base_perms[:, None],
                np.einsum('ik,jki -> ij', diff_ab_perms, rj_d_desc_perms),
            )

            diff_ab_outer_perms -= np.einsum(
                'ijk,k->ji',
                rj_d_desc_perms,
                (sig_pow2 + sig * norm_ab_perms) * mat52_base_perms,
            )

            ri_d_desc = desc_func.d_desc_from_comp(R_d_desc[i, :, :])[0]
            K_block = diff_ab_outer_perms.T.dot(ri_d_desc).T
            diag_K[i*dim_i:(i+1)*dim_i] = np.diag(K_block)

            # K[blk_i, blk_j] = desc_func.vec_dot_d_desc(
            #     R_d_desc[i, :, :], diff_ab_outer_perms.T
            # ).T

            # K[blk_i, blk_j] = desc_func.vec_dot_d_desc(
            #     R_d_desc[i, :, :], diff_ab_outer_perms.T
            # ).T
            # desc_func.vec_dot_d_desc(
            #    R_d_desc[i, :, :], diff_ab_outer_perms.T, out=K[blk_i, blk_j].T
            # )

            # if exploit_sym and (
            #         cols_m_limit is None or i < cols_m_limit
            # ):  # this will never be called with 'keep_idxs_3n' set to anything else than [:]
            #     K[blk_j, blk_i] = K[blk_i, blk_j].T

        if use_E_cstr:
            assert False, 'not implemented yet'
            # E_off = K.shape[0] - n_train, K.shape[1] - n_train
            # blk_j_full = slice(j * dim_i, (j + 1) * dim_i)
            # for i in range(n_train):
            #     diff_ab_perms = R_desc[i, :] - rj_desc_perms
            #     norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
            #
            #     K_fe = (
            #             5
            #             * diff_ab_perms
            #             / (3 * sig ** 3)
            #             * (norm_ab_perms[:, None] + sig)
            #             * np.exp(-norm_ab_perms / sig)[:, None]
            #     )
            #     K_fe = -np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)
            #     K[blk_j_full, E_off[1] + i] = K_fe  # vertical
            #     K[E_off[0] + i, blk_j] = K_fe[keep_idxs_3n]  # lower horizontal
            #
            #     K[E_off[0] + i, E_off[1] + j] = K[E_off[0] + j, E_off[1] + i] = -(
            #             1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            #     ).dot(np.exp(-norm_ab_perms / sig))
        return -diag_K
