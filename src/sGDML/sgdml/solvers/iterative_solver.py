#!/usr/bin/python

# MIT License
#
# Copyright (c) 2020-2021 Stefan Chmiela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import partial
import inspect
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import timeit
import time
import collections

import pickle
from datetime import datetime

import scipy.sparse.linalg

from .. import DONE, NOT_DONE
from ..utils import ui
from ..predict import GDMLPredict

from .iterative_cholesky import Iterative as IterativeCholesky
from . import dev_utils

try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True

from tqdm import tqdm

CG_STEPS_HIST_LEN = (
    100  # number of past steps to consider when calculatating solver effectiveness
)
EFF_RESTART_THRESH = 0  # if solver effectiveness is less than that percentage after 'CG_STEPS_HIST_LEN'-steps, a solver restart is triggert (with stronger preconditioner)
EFF_EXTRA_BOOST_THRESH = (
    50  # increase preconditioner more aggressively below this efficiency threshold
)

glob_U = None
glob_s = None
glob_eigvals_K = None
global_type_delet = None        # which svd preconditioner to be usedd


class CGRestartException(Exception):
    pass


class Iterative(object):
    def __init__(
            self, gdml_train, desc, callback=None, max_processes=None, use_torch=False
    ):

        self.gdml_train = gdml_train
        self.gdml_predict = None
        self.desc = desc

        self.callback = callback

        self._max_processes = max_processes
        self._use_torch = use_torch

        # this will be set once the kernel operator is used on the GPU with pytorch
        self._gpu_batch_size = 0

    # from memory_profiler import profile

    # @profile
    def _init_precon_operator(
            self, task, R_desc, R_d_desc, tril_perms_lin, inducing_pts_idxs, callback=None
    ):

        lam = task['lam']
        lam_inv = 1.0 / lam

        sig = task['sig']

        use_E_cstr = task['use_E_cstr']

        if callback is not None:
            callback = partial(
                callback,
                disp_str='Assembling (partial) kernel matrix',
            )

        K_nm = self.gdml_train._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            self.desc,
            use_E_cstr=use_E_cstr,
            col_idxs=inducing_pts_idxs,
            callback=callback,
        )

        n, m = K_nm.shape
        K_mm = K_nm[inducing_pts_idxs, :]
        print(f'n = {n}, m = {m}\n'
              f'K_nm.shape = {K_nm.shape}')

        if callback is not None:
            callback = partial(
                callback,
                disp_str='Factorizing',
            )
            callback(NOT_DONE)

        # print()
        # print(np.linalg.cond(K_nm))
        # print(np.linalg.cond(K_nm.T.dot(K_nm)))
        # print(np.linalg.cond(K_mm))

        # q,r = np.linalg.qr(np.vstack(K_nm, -lam * K_mm))

        # print(q.shape)

        # r = np.linalg.qr(-K_nm, mode='r')
        # l, lower = sp.linalg.cho_factor(
        #    -K_mm + np.eye(m)*1e-10, overwrite_a=False, check_finite=False
        # )
        # l, lower = self._cho_factor_stable(-K_mm)

        # lu, d, perm = sp.linalg.ldl(-K_mm)
        # lower = True

        # rl = sp.linalg.solve_triangular(
        #       r,
        #       lu,
        #       lower=lower,
        #       trans='T',
        #       overwrite_b=False,
        #       check_finite=False,
        #   )

        # print(np.linalg.cond(rl))

        # print(np.linalg.cond(K_nm))
        # print(np.linalg.cond(K_nm.T.dot(K_nm)))

        # sys.exit()

        # r.T.dot(r)

        # w, _ = np.linalg.eig(rl.dot(rl.T))
        # print(w)

        # l, lower = sp.linalg.cho_factor(
        #    lam * rl.dot(rl.T) + np.eye(m), overwrite_a=False, check_finite=False
        # )

        # l, lower = sp.linalg.cho_factor(
        #    lam * rl.dot(rl.T) + np.eye(m), overwrite_a=False, check_finite=False
        # )

        # lu, d, perm = sp.linalg.ldl(lam * rl.dot(rl.T) + np.eye(m))

        # d = np.linalg.pinv(d)

        # inner = r.T.dot(lam * rl.dot(rl.T) + np.eye(m)).dot(r)

        # L2 = r.T.dot(lu)

        ##Slower but accurate QR solve
        # [Q,~] = qr([Lp./sqrt(dd); eye(kk)], 0);
        # Q1 = Q(1:n,:)./sqrt(dd);
        # P = @(y) y./dd - Q1*(Q1'*y);

        # r = np.linalg.qr(-K_nm, mode='r')

        # q,_ = np.linalg.qr(np.vstack((r.T / np.sqrt(lam), np.eye(m))))

        # q1 = q[:m,:] / np.sqrt(lam)

        # print(r.shape)
        # print(q.shape)
        # print(q1.shape)

        # sys.exit()

        DEBUG_USE_OLD = False

        if DEBUG_USE_OLD:

            inner = -lam * K_mm + K_nm.T.dot(K_nm)
            L, lower = self._cho_factor_stable(inner, lam)

        else:

            # https://stats.stackexchange.com/questions/398865/numerically-stable-sparse-gaussian-process-regression-matrix-inversion

            L_mm, lower = self._cho_factor_stable(-K_mm)
            K_nm = sp.linalg.solve_triangular(
                L_mm,
                K_nm.T,
                lower=lower,
                trans='T',
                overwrite_b=True,
                check_finite=False,
            ).T

            # print(K_nm.shape)

            inner = K_nm.T.dot(K_nm)
            inner[np.diag_indices_from(inner)] += lam

            # u, s, _ = np.linalg.svd(-K_nm.T, full_matrices=False)

            # print(s)
            # sys.exit()

            # L_svd = u * np.sqrt((s**2 + lam))
            # L_svd = u

            # print(L_svd.dot(L_svd.T) - inner)

            # print((s**2 + lam) - np.sqrt((s**2 + lam))**2)

            # print(u.dot(u.T))

            # sys.exit()

            # r = np.linalg.qr(K_nm, mode='r')
            # r[np.diag_indices_from(r)] += np.sqrt(lam)
            # L = r.T
            # lower = True

            L, lower = self._cho_factor_stable(inner)

            # L = Li2
            # K_nm = K_nm_hat


        b_start, b_size = 0, int(n / 10)  # update in percentage steps of 10
        for b_stop in list(range(b_size, n, b_size)) + [n]:

            K_nm[b_start:b_stop, :] = sp.linalg.solve_triangular(
                L,
                K_nm[b_start:b_stop, :].T,
                lower=lower,
                trans='T',
                overwrite_b=True,
                check_finite=False,
            ).T  # Note: Overwrites K_nm to save memory

            if callback is not None:
                callback(b_stop, n)

            b_start = b_stop

        # K_nm = np.linalg.solve(L_svd, K_nm.T).T
        # K_nm = L_svd.T.dot(K_nm.T).T

        # print(L_svd.dot(L_svd.T))
        # sys.exit()

        L_inv_K_mn = K_nm.T

        if self._use_torch and False:  # TURNED OFF!
            _torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            L_inv_K_mn_torch = torch.from_numpy(L_inv_K_mn).to(_torch_device)

        global is_primed
        is_primed = False

        def _P_vec(v):

            global is_primed
            if not is_primed:
                is_primed = True
                return v

            if self._use_torch and False:  # TURNED OFF!

                v_torch = torch.from_numpy(v).to(_torch_device)[:, None]
                return (
                               L_inv_K_mn_torch.t().mm(L_inv_K_mn_torch.mm(v_torch)) - v_torch
                       ).cpu().numpy() * lam_inv

            else:

                # v -= L_inv_K_mn.T.dot(L_inv_K_mn.dot(v))
                # return v * -lam_inv

                # return (L_inv_K_mn.T.dot(L_inv_K_mn.dot(v)) - v) * lam_inv

                # return (L_inv_K_mn.T.dot(L_inv_K_mn.dot(v) * 1.0/(s**2 + lam)) - v) * lam_inv

                ret = L_inv_K_mn.T.dot(L_inv_K_mn.dot(v))
                ret -= v
                ret *= lam_inv
                return ret

                # return (L_inv_K_mn.T.dot(L_inv_K_mn.dot(v)) - v) * lam_inv

        return sp.sparse.linalg.LinearOperator((n, n), matvec=_P_vec)

        # @profile

    def _init_precon_operator_sb(
            self, task, R_desc, R_d_desc, tril_perms_lin, inducing_pts_idxs, callback=None
    ):

        lam = task['lam']
        lam_inv = 1.0 / lam

        sig = task['sig']

        use_E_cstr = task['use_E_cstr']

        if callback is not None:
            callback = partial(
                callback,
                disp_str='Assembling (partial) kernel matrix',
            )

        K_nm = -self.gdml_train._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            self.desc,
            use_E_cstr=use_E_cstr,
            col_idxs=inducing_pts_idxs,
            callback=callback,
        )

        n, m = K_nm.shape
        K_mm = K_nm[inducing_pts_idxs, :]
        print(f'n = {n}, m = {m}\n'
              f'K_nm.shape = {K_nm.shape}')

        if callback is not None:
            callback = partial(
                callback,
                disp_str='Factorizing',
            )
            callback(NOT_DONE)
        # inner = lam * K_mm + K_nm.T @ K_nm          # shape: (m, m)
        # L_inner, lower = self._cho_factor_stable(inner)
        # L_inner = scipy.linalg.cholesky(inner + 1E-16 * np.eye(m), lower=True)
        # P_invers = sp.linalg.solve_triangular(L_inner, K_nm.T, lower=True)

        L_m = scipy.linalg.cholesky(K_mm + 1E-16 * np.eye(m), lower=True)       # shape: (m, m)
        Kbar_nm = sp.linalg.solve_triangular(L_m, K_nm.T, lower=True).T         # shape: (n, m)
        inner = lam * np.eye(m) + Kbar_nm.T @ Kbar_nm           # shape: (m, m)
        L_inner = scipy.linalg.cholesky(inner, lower=True)       # shape: (m, m)
        P_invers = sp.linalg.solve_triangular(L_inner, Kbar_nm.T, lower=True)         # shape: (n, m)

        def apply_inv(a):  # a.shape: N or N x N
            temp_vec = P_invers.T @ (P_invers @ a)
            x = lam_inv * (a - temp_vec)
            return -x

        return sp.sparse.linalg.LinearOperator(shape=(n, n), matvec=apply_inv)

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

            pred -= lam * v
            return pred

        return sp.sparse.linalg.LinearOperator((n, n), matvec=_K_vec)

    def _lev_scores(
            self,
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            lam,
            use_E_cstr,
            n_inducing_pts,
            idxs_ordered_by_lev_score=None,
            # importance ordering of columns used to pick the columns to approximate leverage scoresid (optional)
            callback=None,
    ):

        n_train, dim_d = R_d_desc.shape[:2]
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)

        # Convert from training points to actual columns.
        # dim_m = n_inducing_pts * dim_i
        dim_m = np.maximum(1,
                           n_inducing_pts // 4) * dim_i  # only use 1/4 of inducing points for leverage score estimate

        # Which columns to use for leverage score approximation?
        if idxs_ordered_by_lev_score is None:
            lev_approx_idxs = np.sort(
                np.random.choice(n_train * dim_i, dim_m, replace=False))  # random subset of columns
            # lev_approx_idxs = np.s_[
            #    :dim_m
            # ]  # first 'dim_m' columns (faster kernel construction)
        else:
            assert len(idxs_ordered_by_lev_score) == n_train * dim_i

            lev_approx_idxs = np.sort(
                idxs_ordered_by_lev_score[-dim_m:]
            )  # choose 'dim_m' columns according to provided importance ordering

        if callback is not None:
            callback = partial(
                callback,
                disp_str='Approx. leverage scores (1/3 assembling matrix)',
            )

        K_nm = self.gdml_train._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            self.desc,
            use_E_cstr=use_E_cstr,
            col_idxs=lev_approx_idxs,
            callback=callback,
        )
        K_mm = K_nm[lev_approx_idxs, :]

        if callback is not None:
            callback = partial(
                callback, disp_str='Approx. leverage scores (2/3 factoring)'
            )
            callback(NOT_DONE)

        L, lower = self._cho_factor_stable(-K_mm)

        callback(DONE)

        if callback is not None:
            callback = partial(
                callback, disp_str='Approx. leverage scores (3/3 constructing)'
            )

        n = K_nm.shape[0]
        b_start, b_size = 0, int(n / 10)  # update in percentage steps of 10
        for b_stop in list(range(b_size, n, b_size)) + [n]:

            K_nm[b_start:b_stop, :] = sp.linalg.solve_triangular(
                L,
                K_nm[b_start:b_stop, :].T,
                lower=lower,
                trans='T',
                overwrite_b=True,
                check_finite=False,
            ).T  # Note: Overwrites K_nm to save memory

            if callback is not None:
                callback(b_stop, n)

            b_start = b_stop
        B = K_nm.T

        B_BT_lam = B.dot(B.T)
        B_BT_lam[np.diag_indices_from(B_BT_lam)] += lam

        # Leverage scores for all columns.
        # lev_scores = np.einsum('ij,ij->j', B, np.linalg.solve(B_BT_lam, B))

        # Leverage scores for all columns.
        # C, C_lower = sp.linalg.cho_factor(
        #    B_BT_lam, overwrite_a=True, check_finite=False
        # )
        C, C_lower = self._cho_factor_stable(B_BT_lam)
        B = sp.linalg.solve_triangular(
            C, B, lower=C_lower, trans='T', overwrite_b=True, check_finite=False
        )
        C_B = B
        lev_scores = np.einsum('i...,i...->...', C_B, C_B)

        return lev_scores, np.argsort(lev_scores)

    # performs a cholesky decompostion of a matrix, but regularizes the matrix (if neeeded) until its positive definite
    def _cho_factor_stable(self, M, min_eig=None):
        """
        Performs a Cholesky decompostion of a matrix, but regularizes
        as needed until its positive definite.

        Parameters
        ----------
            M : :obj:`numpy.ndarray`
                Matrix to factorize.
            min_eig : float
                Force lowest eigenvalue to
                be a certain (positive) value
                (default: machine precision)

        Returns
        -------
            :obj:`numpy.ndarray`
                Matrix whose upper or lower triangle contains the Cholesky factor of a. Other parts of the matrix contain random data.
            boolean
                Flag indicating whether the factor is in the lower or upper triangle
        """
        if True:
            lo_eig = sp.linalg.eigh(M, eigvals_only=True, eigvals=(0, 0))
            sgn = 1 if lo_eig <= 0 else -1
            M[np.diag_indices_from(M)] += sgn * 1.e-15  # delete smallest eigenvalues
            L, lower = sp.linalg.cho_factor(
                M, overwrite_a=False, check_finite=False
            )
            return L, lower
        eps = np.finfo(float).eps
        eps_mag = int(np.floor(np.log10(eps)))

        if min_eig is None:
            min_eig = eps
        else:
            assert min_eig > 0

        lo_eig = sp.linalg.eigh(M, eigvals_only=True, eigvals=(0, 0))
        print(f'_cho_factor_stable, lo_eig: {lo_eig}')
        if lo_eig < min_eig:
            sgn = 1 if lo_eig <= 0 else -1
            lam = sgn * (min_eig - lo_eig)     # delete smallest eigenvalues
            print(f'Stable cho factorization with lam: {lam:.2E}')
            M[np.diag_indices_from(M)] += lam

        for reg in 10.0 ** np.arange(
                eps_mag, 2
        ):  # regularize more and more aggressively (strongest regularization: 1)
            try:

                L, lower = sp.linalg.cho_factor(
                    M, overwrite_a=False, check_finite=False
                )

            except np.linalg.LinAlgError as e:

                if 'not positive definite' in str(e):
                    print(f'Stable cho factorization additional reg: {reg:.2E}')
                    M[np.diag_indices_from(M)] += reg
                else:
                    raise e
            else:

                return L, lower

    def solve(
            self,
            task,
            R_desc,
            R_d_desc,
            tril_perms_lin,
            y,
            y_std,
            save_progr_callback=None,
            break_percentage=None,
            str_preconditioner='',
            flag_eigvals=False,
    ):
        start_solve_routine = timeit.default_timer()
        global num_iters, start, resid, avg_tt, m

        n_train, n_atoms = task['R_train'].shape[:2]
        dim_i = 3 * n_atoms
        n = 3 * n_train * n_atoms       # kernel size

        sig = task['sig']
        lam = task['lam']

        # these keys are only present if the task was created from an existing model
        alphas0_F = task['alphas0_F'] if 'alphas0_F' in task else None
        alphas0_E = task['alphas0_E'] if 'alphas0_E' in task else None
        num_iters0 = task['solver_iters'] if 'solver_iters' in task else 0

        import copy
        if break_percentage is None:
            n_inducing_pts_init = copy.deepcopy(task['n_inducing_pts_init'])
        else:
            n_inducing_pts_init = int(max(np.ceil(break_percentage * n_train), 1))

        if 'inducing_pts_idxs' in task:  # only if task was created from model
            n_inducing_pts_init = len(task['inducing_pts_idxs']) // (3 * n_atoms)

        # How many inducing points to use (for Nystrom approximation, as well as the approximation of leverage scores).
        # Note: this number is automatically increased if necessary.
        n_inducing_pts = min(n_train, n_inducing_pts_init)

        if self.callback is not None:
            self.callback = partial(
                self.callback,
                disp_str='Constructing preconditioner',
            )
        subtask_callback = partial(ui.sec_callback, main_callback=self.callback)

        start_preconditioner = timeit.default_timer()
        start_cholesky = timeit.default_timer()
        start = timeit.default_timer()

        idxs_ordered_by_lev_score = None
        # set-up preconditioner
        lev_scores_keys = ['lev_scores', 'random_scores', 'inverse_lev', 'lev_random', 'truncated_cholesky',
                           'truncated_cholesky_custom', 'rank_k_lev_scores', 'rank_k_lev_scores_custom']
        if np.sum([str_preconditioner == lev_key for lev_key in lev_scores_keys]) == 1:
            k = int(break_percentage * n)
            if 'inducing_pts_idxs' in task:
                inducing_pts_idxs = task['inducing_pts_idxs']
                assert False, 'Nor applicable in this setting'
            else:
                # Determine good inducing points.
                if str_preconditioner == 'random_scores':
                    print('Random uniform columns used')
                    inducing_pts_idxs = np.random.choice(np.arange(n), size=k, replace=False)
                    inducing_pts_idxs = np.sort(inducing_pts_idxs)
                elif str_preconditioner in ['truncated_cholesky', 'truncated_cholesky_custom']:
                    print('Truncated cholesky decomposition')
                    k_truncate = task['truncated_cholesky']
                    k_truncate = k_truncate if k_truncate < k else k

                    iterative_cholesky = IterativeCholesky(gdml_train=self.gdml_train, desc=self.desc, task=task,
                                                           callback=self.callback,
                                                           max_processes=self._max_processes, use_torch=self._use_torch)

                    diag_K = iterative_cholesky._assemble_kernel_mat_diag(tril_perms_lin=tril_perms_lin,
                                                                          sig=task['sig'],
                                                                          R_desc=R_desc, R_d_desc=R_d_desc, n=n)
                    K_op = self._init_kernel_operator(
                        task, R_desc, R_d_desc, tril_perms_lin, lam, n, callback=subtask_callback
                    )
                    P_op, info_cholesky = iterative_cholesky._init_precon_operator(diag_K, -K_op,
                                                                                   lam_regularization=task['lam'],
                                                                                   break_percentage=float(k_truncate/n))
                    # inducing_pts_idxs = np.random.choice(np.arange(n), size=k, replace=False)
                    index_columns_cholesky = info_cholesky['index_columns']
                    inducing_pts_cholesky = index_columns_cholesky[:k_truncate]
                    k_random = int(k-k_truncate) if k_truncate < k else 0
                    inducing_pts_random = np.random.choice(index_columns_cholesky[k_truncate:],
                                                           size=k_random, replace=False)
                    inducing_pts_idxs = np.concatenate([inducing_pts_cholesky, inducing_pts_random])
                    inducing_pts_idxs = np.sort(inducing_pts_idxs)
                elif str_preconditioner in ['rank_k_lev_scores', 'rank_k_lev_scores_custom']:
                    leverage_scores = self._rank_k_leverage_scores(task,
                                                                     R_desc,
                                                                     R_d_desc,
                                                                     tril_perms_lin,
                                                                     break_percentage,
                                                                     callback=subtask_callback,)
                    lev_scores_normalized = leverage_scores / leverage_scores.sum()
                    inducing_pts_idxs = np.random.choice(np.arange(n), size=k, replace=False, p=lev_scores_normalized)
                    inducing_pts_idxs = np.sort(inducing_pts_idxs)

                elif str_preconditioner in ['lev_scores', 'inverse_lev', 'lev_random']:
                    lev_scores, idxs_ordered_by_lev_score = self._lev_scores(
                        R_desc,
                        R_d_desc,
                        tril_perms_lin,
                        sig,
                        lam,
                        False,  # use_E_cstr
                        n_inducing_pts,
                        callback=subtask_callback,
                    )

                    if str_preconditioner == 'inverse_lev':
                        print('Inverse deterministic leverage scores')
                        inducing_pts_idxs = np.sort(idxs_ordered_by_lev_score[:k])

                    elif str_preconditioner == 'lev_scores':
                        print('Deterministic leverage scores')
                        inducing_pts_idxs = np.sort(idxs_ordered_by_lev_score[-k:])

                    elif str_preconditioner == 'lev_random':
                        print('Probabilistic leverage scores ')
                        lev_scores_normalized = lev_scores/lev_scores.sum()
                        inducing_pts_idxs = np.random.choice(np.arange(n), size=k, replace=False, p=lev_scores_normalized)

                        # lev_scores_rescaled = lev_scores_normalized**5
                        # lev_scores_rescaled /= lev_scores_rescaled.sum()
                        # inducing_pts_idxs = np.random.choice(np.arange(n), size=k, replace=False, p=lev_scores_rescaled)

                        inducing_pts_idxs = np.sort(inducing_pts_idxs)
                    else:
                        assert False, f'Something went wrong with str_preconditioner = {str_preconditioner} inside lev'
                else:
                    raise ValueError(f'Something went wrong with str_perconditioner = {str_preconditioner}.')

                assert inducing_pts_idxs.shape == (k,), 'Incorrect number of inducing points.'
                if str_preconditioner in ['truncated_cholesky_custom', 'rank_k_lev_scores_custom']:
                    P_op = self._init_precon_operator_sb(
                        task,
                        R_desc,
                        R_d_desc,
                        tril_perms_lin,
                        inducing_pts_idxs,
                        callback=subtask_callback,
                    )
                else:
                    P_op = self._init_precon_operator(
                        task,
                        R_desc,
                        R_d_desc,
                        tril_perms_lin,
                        inducing_pts_idxs,
                        callback=subtask_callback,
                    )

        elif str_preconditioner == 'cholesky':
            print('Incomplete Cholesky ')
            iterative_cholesky = IterativeCholesky(gdml_train=self.gdml_train, desc=self.desc, task=task, callback=self.callback,
                                                   max_processes=self._max_processes, use_torch=self._use_torch)

            diag_K = iterative_cholesky._assemble_kernel_mat_diag(tril_perms_lin=tril_perms_lin, sig=task['sig'],
                                                                  R_desc=R_desc, R_d_desc=R_d_desc, n=n)
            K_op = self._init_kernel_operator(
                task, R_desc, R_d_desc, tril_perms_lin, lam, n, callback=subtask_callback
            )
            P_op, info_cholesky = iterative_cholesky._init_precon_operator(diag_K, -K_op, lam_regularization=task['lam'],
                                                                           break_percentage=break_percentage)

            inducing_pts_idxs = np.arange(int(break_percentage * K_op.shape[0]))        # used to measure k outside this function

        elif str_preconditioner == 'eigvec_precon' or str_preconditioner == 'eigvec_precon_block_diagonal' \
                or str_preconditioner == 'eigvec_precon_atomic_interactions':
            P_op, k_final_precon = self._init_precon_operator_eigvals(
                task,
                R_desc,
                R_d_desc,
                tril_perms_lin,
                break_percentage,
                callback=subtask_callback,
            )
            inducing_pts_idxs = np.arange(k_final_precon)

        else:
            raise NotImplementedError(f'str_preconditioner = {str_preconditioner}')

        stop = timeit.default_timer()
        stop_preconditioner = timeit.default_timer()
        total_time_cholesky = stop_preconditioner - start_cholesky
        total_time_preconditioner = stop_preconditioner - start_preconditioner
        print(f'time preconditioner\n stop - start: {total_time_cholesky:.1E}s, {total_time_cholesky / 60:.2f}min')

        if self.callback is not None:
            dur_s = stop - start
            sec_disp_str = 'took {:.1f} s'.format(dur_s) if dur_s >= 0.1 else ''
            self.callback(DONE, sec_disp_str=sec_disp_str)

        if self.callback is not None:
            self.callback = partial(
                self.callback,
                disp_str='Initializing solver',
            )
        subtask_callback = partial(ui.sec_callback, main_callback=self.callback)

        n = P_op.shape[0]
        K_op = self._init_kernel_operator(
            task, R_desc, R_d_desc, tril_perms_lin, lam, n, callback=subtask_callback
        )

        num_iters = num_iters0

        if task['use_E_cstr'] and self._use_torch:
            print('NOT IMPLEMENTED!!!')
            sys.exit()

        if self.callback is not None:

            num_devices = (
                mp.cpu_count() if self._max_processes is None else self._max_processes
            )
            if self._use_torch:
                num_devices = (
                    torch.cuda.device_count()
                    if torch.cuda.is_available()
                    else torch.get_num_threads()
                )
            hardware_str = '{:d} {}{}{}'.format(
                num_devices,
                'GPU' if self._use_torch and torch.cuda.is_available() else 'CPU',
                's' if num_devices > 1 else '',
                '[PyTorch]' if self._use_torch else '',
            )

            self.callback(NOT_DONE, sec_disp_str=None)

        start = 0
        resid = 0
        avg_tt = 0

        global alpha_t, eff, steps_hist, callback_disp_str

        alpha_t = None
        steps_hist = collections.deque(
            maxlen=CG_STEPS_HIST_LEN
        )  # moving average window for step history

        increase_ip = False

        global pbar
        callback_disp_str = 'Initializing solver'

        def _cg_status(xk):
            global num_iters, start, resid, alpha_t, avg_tt, m, eff, steps_hist, callback_disp_str
            global pbar
            pbar.update(1)
            stop = timeit.default_timer()
            tt = 0.0 if start == 0 else (stop - start)
            avg_tt += tt
            start = timeit.default_timer()

            old_resid = resid
            resid = inspect.currentframe().f_back.f_locals['resid']
            pbar.set_postfix_str(f'resid = {resid:.2E}')

            step = 0 if num_iters == 0 else resid - old_resid
            steps_hist.append(step)

            steps_hist_arr = np.array(steps_hist)
            steps_hist_all = np.abs(steps_hist_arr).sum()
            steps_hist_ratio = (
                (-steps_hist_arr.clip(max=0).sum() / steps_hist_all)
                if steps_hist_all > 0
                else 1
            )
            eff = 0 if num_iters == 0 else (int(100 * steps_hist_ratio) - 50) * 2

            if tt > 0.0 and num_iters % int(np.ceil(1.0 / tt)) == 0:  # once per second

                train_rmse = resid / np.sqrt(len(y))

                if self.callback is not None:
                    callback_disp_str = 'Training error (RMSE): forces {:.4f}'.format(train_rmse)

                    self.callback(
                        NOT_DONE,
                        disp_str=callback_disp_str,
                        sec_disp_str=(
                            '{:d} iter @ {} iter/s [eff: {:d}%] k: {:d}'.format(
                                num_iters,
                                '{:.1f}'.format(1.0 / tt),
                                eff,
                                n_inducing_pts,
                            )
                        ),
                    )

            # Write out current solution as a model file once every 2 minutes (give or take).
            if tt > 0.0 and num_iters % int(np.ceil(2 * 60.0 / tt)) == 0:

                # TODO: support for +E constraints (done?)
                alphas_F, alphas_E = -xk, None
                if task['use_E_cstr']:
                    alphas_F, alphas_E = -xk[:-n_train], -xk[-n_train:]

                unconv_model = self.gdml_train.create_model(
                    task,
                    'cg',
                    R_desc,
                    R_d_desc,
                    tril_perms_lin,
                    y_std,
                    alphas_F,
                    alphas_E=alphas_E,
                    solver_resid=resid,
                    solver_iters=num_iters + 1,
                    norm_y_train=np.linalg.norm(y),
                    inducing_pts_idxs=inducing_pts_idxs,
                )

                # recover integration constant
                n_train = task['E_train'].shape[0]
                R = task['R_train'].reshape(n_train, -1)

                self.gdml_predict.set_alphas(R_d_desc, alphas_F, alphas_E=alphas_E)
                E_pred, _ = self.gdml_predict.predict(R)
                E_pred *= y_std
                E_ref = np.squeeze(task['E_train'])

                unconv_model['c'] = np.sum(E_ref - E_pred) / E_ref.shape[0]

                if save_progr_callback is not None:
                    save_progr_callback(unconv_model)

            num_iters += 1

            n_train = task['E_train'].shape[0]
            if (
                    len(steps_hist) == CG_STEPS_HIST_LEN
                    and eff <= EFF_RESTART_THRESH
                    and n_inducing_pts < n_train
            ):
                alpha_t = xk
                # raise CGRestartException

        pbar = tqdm(total=y.shape[0]*10, desc='CG')

        alphas0 = None
        # alphas0 = np.random.uniform(low=-1, high=1, size=y.shape)
        if alphas0_F is not None:  # TODO: improve me: this iwll not workt with E_cstr
            alphas0 = -alphas0_F

        if alphas0_E is not None:
            alphas0_E *= -1  # TODO: is this correct (sign)?
            alphas0 = np.hstack((alphas0, alphas0_E))

        if flag_eigvals is True:
            eigvals = dev_utils.get_eigvals(K_op, P_op)
            global glob_eigvals_K
            if glob_eigvals_K is None:
                print('Compute raw eigvals_K from scratch')
                identity_fn = lambda vec: vec
                unity_op = scipy.sparse.linalg.LinearOperator(shape=K_op.shape, matvec=identity_fn)
                glob_eigvals_K = dev_utils.get_eigvals(K_op, unity_op)
                eigvals_K = glob_eigvals_K
            else:
                print('reuse previous eigvals_K')
                eigvals_K = glob_eigvals_K

        num_restarts = 0
        while True:
            try:
                tic_start = timeit.default_timer()
                alphas, info = sp.sparse.linalg.cg(
                    -K_op,
                    y,
                    x0=alphas0 if alpha_t is None else alpha_t,
                    M=P_op,
                    tol=task['solver_tol'],  # norm(residual) <= max(tol*norm(b), atol)
                    atol=None,
                    maxiter=3 * n_atoms * n_train * 5 if flag_eigvals is False else 10,
                    # allow 10x as many iterations as theoretically needed (at perfect precision)
                    callback=_cg_status,
                )
                toc_stop = timeit.default_timer()
                total_time_cg = toc_stop - tic_start
                pbar.close()
                alphas = -alphas

            except CGRestartException:

                num_restarts += 1
                steps_hist.clear()

                n_inducing_pts += (
                    5 if eff <= EFF_EXTRA_BOOST_THRESH else 1
                )  # increase more agressively if convergence is especially weak
                n_inducing_pts = min(n_inducing_pts, n_train)

                subtask_callback = partial(ui.sec_callback,
                                           main_callback=partial(self.callback, disp_str=callback_disp_str))

                # NEW
                # Pause multiprocessing pool

                # num_workers = self.gdml_predict.num_workers
                # self.gdml_predict._set_num_workers()
                # NEW

                if (
                        num_restarts == 1 or num_restarts % 10 == 0 or idxs_ordered_by_lev_score is None
                ):  # recompute leverate scoresid on first restart (first approximation is bad) and every 10 restarts.

                    # Use leverage scoresid from last run to estimate better ones this time.
                    idxs_ordered_by_lev_score = self._lev_scores(
                        R_desc,
                        R_d_desc,
                        tril_perms_lin,
                        sig,
                        lam,
                        False,  # use_E_cstr
                        n_inducing_pts,
                        idxs_ordered_by_lev_score=idxs_ordered_by_lev_score,
                        callback=subtask_callback,
                    )

                dim_m = n_inducing_pts * dim_i
                inducing_pts_idxs = np.sort(
                    idxs_ordered_by_lev_score[-dim_m:]
                )

                del P_op
                P_op = self._init_precon_operator(
                    task,
                    R_desc,
                    R_d_desc,
                    tril_perms_lin,
                    inducing_pts_idxs,
                    callback=subtask_callback,
                )

                # NEW
                # Restart multiprocessing pool
                # self.gdml_predict._set_num_workers(num_workers)
                # NEW

            else:
                break

        is_conv = info == 0
        print(f'inducing_pts_idxs.shape = {inducing_pts_idxs.shape}')

        if self.callback is not None:
            is_conv_warn_str = '' if is_conv else ' (NOT CONVERGED)'
            self.callback(
                DONE,
                disp_str='Training on {:,} points{}'.format(n_train, is_conv_warn_str),
                sec_disp_str=(
                    '{:d} iter @ {} iter/s'.format(
                        num_iters,
                        '{:.1f}'.format(num_iters / avg_tt) if avg_tt > 0 else '--',
                    )
                ),
                done_with_warning=not is_conv,
            )

        end_solve_routine = timeit.default_timer()
        total_time_solve = end_solve_routine - start_solve_routine
        print(f'\n==================================================================================\n'
              f'total time solve:  {datetime.strftime(datetime.now(), "%c")} \n'
              f'{total_time_solve:.1E}s, {total_time_solve / 60:.2f}min \n'
              f'==================================================================================\n')
        info_iterative_solver = {'is_conv': is_conv,
                                 'total_time_cholesky': total_time_cholesky,
                                 'total_time_cg': total_time_cg,
                                 'total_time_solve': total_time_solve,
                                 'total_time_preconditioner': total_time_preconditioner}

        if flag_eigvals is True:
            info_iterative_solver['eigvals'] = eigvals
            info_iterative_solver['eigvals_K'] = eigvals_K

        if str_preconditioner == 'cholesky':
            info_iterative_solver.update(info_cholesky)

        train_rmse = resid / np.sqrt(len(y))
        return alphas, num_iters, resid, train_rmse, inducing_pts_idxs, is_conv, info_iterative_solver

    def _rank_k_leverage_scores(
            self, task, R_desc, R_d_desc, tril_perms_lin, break_percentage, callback=None
    ) -> np.ndarray:
        """
        Expand use a small eigenvalue decomposition to cover the complete preconditioner in a periodic fashion
        """

        lam = task['lam']
        sig = task['sig']
        use_E_cstr = task['use_E_cstr']

        if callback is not None:
            callback = partial(
                callback,
                disp_str='Assembling (partial) kernel matrix',
            )

        n = task['F_train'].size  # matrix size K
        k = np.max([int(break_percentage * n), 1])

        # k = n_inducing_pts*molecule_size
        # n_inducing_pts = int(np.min([4*k, n]))
        # n_inducing_pts = n
        # inducing_pts_idxs = np.arange(n_inducing_pts)

        # TODO: only create quadractic kernel matrix K_kk
        K_nm = self.gdml_train._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            self.desc,
            use_E_cstr=use_E_cstr,
            col_idxs=np.arange(n),
            callback=callback,
        )
        K = K_nm[:n, :n].copy()

        global global_type_delet
        global glob_U
        global glob_s
        type_delete = task['str_preconditioner']
        if global_type_delet != type_delete:
            glob_U = None
            glob_s = None
            global_type_delet = type_delete

        # raw eigen decomposition
        if glob_U is None or glob_s is None:
            # if True:
            print('Compute SVD from scratch')
            U, s, V = sp.linalg.svd(K)  # -np.diag(lam*np.ones(K.shape[0]))
            glob_U = U
            glob_s = s
        else:
            print('reuse previous SVD')
            U = glob_U
            s = glob_s

        # see Def. 1 in https://arxiv.org/pdf/2201.07017.pdf
        # for numerical stability it might be interesting to first calculate the regularization lambda corresponding to
        # k and then use this to calculate the ridge leverage score
        U_k = U[:, :k]
        leverage_scores = np.linalg.norm(U_k, axis=1)

        return leverage_scores

    def _init_precon_operator_eigvals(
            self, task, R_desc, R_d_desc, tril_perms_lin, break_percentage, callback=None
    ):
        """
        Expand use a small eigenvalue decomposition to cover the complete preconditioner in a periodic fashion
        """

        lam = task['lam']
        lam_inv = 1.0 / lam

        sig = task['sig']

        use_E_cstr = task['use_E_cstr']

        if callback is not None:
            callback = partial(
                callback,
                disp_str='Assembling (partial) kernel matrix',
            )

        # n_train = R_desc.shape[0]
        molecule_size = self.desc.dim_i
        n_train, n_atoms, d_space = task['F_train'].shape
        n = task['F_train'].size    # matrix size K
        k = molecule_size * np.max([int(break_percentage * n_train), 1])
        k = np.max([int(break_percentage * n), 1])
        import copy
        k_final = copy.copy(k)
        # break_percentage_actually_used = k/n

        # k = n_inducing_pts*molecule_size
        # n_inducing_pts = int(np.min([4*k, n]))
        n_inducing_pts = n
        inducing_pts_idxs = np.arange(n_inducing_pts)

        # TODO: only create quadractic kernel matrix K_kk
        K_nm = self.gdml_train._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            self.desc,
            use_E_cstr=use_E_cstr,
            col_idxs=inducing_pts_idxs,
            callback=callback,
        )
        K = K_nm[:n_inducing_pts, :n_inducing_pts].copy()
        #
        # # k = int(0.2*n)
        # print(f'n = {n}\n'
        #       f'k = {k}')
        #
        # delete small entries but keep correlations between equivalent atoms
        global global_type_delet
        global glob_U
        global glob_s
        type_delete = task['str_preconditioner']
        if global_type_delet != type_delete:
            glob_U = None
            glob_s = None
            global_type_delet = type_delete
        if type_delete == 'eigvec_precon_atomic_interactions':         # same as building atomic-block-diagonal preconditioner
            abs_maximum = np.abs(K_nm).max()
            eps = 1 * abs_maximum
            mask_atomic_interactions = np.abs(K_nm) < eps
            mask_molecule = np.zeros((molecule_size, molecule_size), dtype=np.bool)
            for i_atom in range(n_atoms):       # defines a block-diagonal matrix with 3x3 blocks
                mask_molecule[3*i_atom:3*(i_atom+1), 3*i_atom:3*(i_atom+1)] = True

            # broadcast mask_molecule to K.shape
            mask_temp = np.concatenate([mask_molecule for _ in range(n_train)])
            mask_trainingpoints = np.concatenate([mask_temp for _ in range(n_train)], axis=1)
            mask_atomic_interactions[mask_trainingpoints] = False       # do not allow to delete self-interaction between a single atom

            mask_oddparity = mask_atomic_interactions != mask_atomic_interactions.T
            assert mask_oddparity.sum() == 0, 'only allow symmetric deletes'
            K[mask_atomic_interactions] = 0
            print(f'percentage of deleted entries: {mask_atomic_interactions.sum()/mask_atomic_interactions.size*100} ==================================================')
        elif type_delete == 'eigvec_precon_block_diagonal':
            mask = np.ones((molecule_size, molecule_size), dtype=bool)
            from scipy.linalg import block_diag
            mask_block_diag = block_diag(*([mask]*n_train))
            mask = np.invert(mask_block_diag)
            # K[np.invert(mask_block_diag)] = 0
            K[np.ones_like(K, dtype=bool)] = 0
            print(
                f'percentage of deleted entries: {mask.sum() / mask.size * 100} ==================================================')
        elif type_delete == 'eigvec_precon':
            print('Standard')
            pass
        else:
            raise NotImplementedError(f'type_delete = {type_delete}')

        store_ridge_leverage_scores = False
        if store_ridge_leverage_scores:
            dict_ridge_lev_scores = {'molecule_size': molecule_size, 'n_atoms': n_atoms, 'n_train': n_train}
            info_keys = ['dataset_name', 'sig', 'lam', 'solver_tol', 'z']
            for label in info_keys:
                dict_ridge_lev_scores[label] = task[label]

            # for lam in np.geomspace(1e-6, 1e-13, 5):
            list_lambda = [1e-5, 1e-8, 1e-10, 1e-12]
            for lam_temp in list_lambda:
                K_lam = K.copy() @ K.T.copy() + lam_temp * np.eye(n)
                temp = sp.linalg.solve(K_lam, K)
                tau = np.diag(K.T @ temp)
                # U, s, V = sp.linalg.svd(K_lam)
                # ridge_lev_scores = np.linalg.norm(U, axis=1)**2
                dict_ridge_lev_scores[f'tau_{lam_temp}'] = tau
            dict_ridge_lev_scores['list_lambda'] = list_lambda
            pickle.dump(dict_ridge_lev_scores, open(f'ridge_levscores_{task["dataset_name"]}_{3*n_atoms*n_train}', 'wb'))
            assert False, 'Created Ridge Leverage score dataset'


        # raw eigen decomposition
        flag_ridge_svg_preonditioner = False
        if flag_ridge_svg_preonditioner is False:       # this is the standard choice
            if glob_U is None or glob_s is None:
            # if True:
                print('Compute SVD from scratch')
                U, s, V = sp.linalg.svd(K)          # -np.diag(lam*np.ones(K.shape[0]))
                glob_U = U
                glob_s = s
            else:
                print('reuse previous SVD')
                U = glob_U
                s = glob_s

            # K_fully_masked = K_nm.copy()
            # K_fully_masked[mask] = 0
            # Ufully, sfully, Vfully = sp.linalg.svd(K_fully_masked)          # -np.diag(lam*np.ones(K.shape[0]))
            # L = (Ufully @ np.diagflat(np.sqrt(sfully)))[:, :1000]
            # Khat = Ufully[:, :k] @ sfully[:k] @ Vfully
            # Khat = L @ L.T

            # Unm, snm, Vnm = sp.linalg.svd(K_nm)          # -np.diag(lam*np.ones(K.shape[0]))
            def svd_preconditioner(U_eigenvectors, s_eigenvalues, k_rank):
                n = len(s_eigenvalues)
                L = (U_eigenvectors @ np.diagflat(np.sqrt(s_eigenvalues)))[:, :k_rank]

                # precompute inner matrices
                kernel = lam * np.eye(k_rank) + (L.T @ L)
                L2 = sp.linalg.cholesky(kernel, lower=True)  # k x k
                temp_mat = sp.linalg.solve_triangular(L2, L.T, lower=True)  # k x N

                def apply_inv(a):  # a.shape: N or N x N
                    temp_vec = temp_mat.T @ (temp_mat @ a)
                    x = lam_inv * (a - temp_vec)
                    return x
                lin_inv_operator = sp.sparse.linalg.LinearOperator(shape=(n, n), matvec=apply_inv)
                return lin_inv_operator

            lin_inv_operator = svd_preconditioner(U_eigenvectors=U, s_eigenvalues=s, k_rank=k)

        else:       # ridge SVD         does not work yet. cannot invert a rank k matrix with n x n dimensions
            # K is negative semi definite
            K_ridge = K - lam * np.eye(n)
            U_ridge, s_ridge, V_ridge = sp.linalg.svd(K_ridge)  # -np.diag(lam*np.ones(K.shape[0]))

            # k = int(0.8 * n)

            K_ridge_k = U_ridge[:, :k] @ np.diag(s_ridge[:k]) @ V_ridge[:k, :]

            print(f'distance to original matrix: {np.linalg.norm(K_ridge - K_ridge_k)}')
            # L = sp.linalg.cholesky(-K_ridge_k)


            def apply_inv(a):  # a.shape: N or N x N
                x = np.linalg.solve(K_ridge_k, a)
                return x

            lin_inv_operator = sp.sparse.linalg.LinearOperator(shape=(n, n), matvec=apply_inv)

        # if k < n:
        #     print('Return unity as preconditioner')
        #     def apply_inv(a):  # a.shape: N or N x N
        #         return a
        #     lin_inv_operator = sp.sparse.linalg.LinearOperator(shape=(n, n), matvec=apply_inv)
        # ==============================================================================================================
        # DEBUG CODE
        # ==============================================================================================================
        import sys
        gettrace = sys.gettrace()

        # For debugging
        debug_status = True if gettrace else False
        if debug_status is True:
            import matplotlib.pyplot as plt
            def plt_mat(mat):
                plt.figure()
                vmax = mat.max()
                plt.imshow(mat, cmap="seismic", vmax=vmax, vmin=-vmax)
                plt.colorbar()
                plt.tight_layout()
                plt.grid(False)

            # Generate label vector.
            E_train_mean = None
            y = task['F_train'].ravel().copy()
            if task['use_E'] and task['use_E_cstr']:
                E_train = task['E_train'].ravel().copy()
                E_train_mean = np.mean(E_train)

                y = np.hstack((y, -E_train + E_train_mean))
                # y = np.hstack((n*Ft, (1-n)*Et))
            y_std = np.std(y)
            y /= y_std

            # # compute relevant dimension with projection on eigvals
            # alphas, info = sp.sparse.linalg.cg(
            #     K + np.diag(np.ones(n))*lam,
            #     -y,
            #     M=lin_inv_operator,
            # )
            # assert info==0, 'training did not converge'
            #
            # U, s, V = sp.linalg.svd(K+ np.diag(np.ones(n))*lam)  # -np.diag(lam*np.ones(K.shape[0]))

            # projection = U.T @ alphas
            # plt.figure('alphas projection')
            # plt.plot(np.abs(projection))
            # plt.semilogy()
            #
            # plt.figure('alphas projection rescaled')
            # plt.plot(np.abs(projection)*np.sqrt(s))
            # plt.semilogy()
            #
            # plt.figure('alphas projection without log')
            # plt.plot(projection)
            #
            # plt.figure('inverse alphas projection')
            # plt.plot(1/np.abs(projection))
            # plt.semilogy()

            # On relevant dimensions in kernel feature spaces, Braun, Buhmann and Muller
            # see sec. 4.1.1
            plt.figure('kernel PCA components')
            plt.plot(np.abs(U.T @ y), label='relevance coefficients')
            plt.plot(s, label='eigenvalues')
            plt.semilogy()

            # plt.figure('eigvals')
            # plt.plot(s)
            # plt.semilogy()
            # np.cumsum(U[0])
            #
            # plt_mat(K)
            #
            # index = np.random.choice(100, 3, replace=False)
            #
            # plt.figure()
            # for i in index:
            #     plt.plot(np.cumsum(np.abs(U[i])), label=i)
            # plt.legend()
            #
            # plt_mat(U)

            # ==============================================================================================================
            # # paper summary plot: matrix, eigenvectors, spectrum, INPUT: kernel K
            from tools import init_plt as c_plt
            c_plt.update_rcParams(half_size_image=True)


            row_max = 1*molecule_size
            col_max = 25*molecule_size
            col_max = n
            K_small = K[:col_max, :col_max].copy()
            Usmall, ssmall, Vsmall = sp.linalg.svd(K_small)


            mat = K_small
            plt.figure('kernel matrix', figsize=(3., 3.))
            vmax = mat.max()
            plt.imshow(mat, cmap="seismic", vmax=vmax, vmin=-vmax)
            # plt.colorbar()
            plt.grid(False)
            plt.tick_params(left=False,
                            bottom=False,
                            labelleft=False,
                            labelbottom=False)
            plt.tight_layout(pad=0.1)

            fig = plt.figure('kernel spectrum')
            ax = plt.subplot(1, 1, 1)
            ax.plot(s, label='eigenvalues')
            # ax.plot(ssmall, label='eigenvalues')
            plt.plot(np.abs(U.T @ y), label='kernel PCA coefficients', linewidth=0.5, alpha=0.5)
            ax.semilogy()
            ax.tick_params(left=True,
                           bottom=False,
                           labelleft=True,
                           labelbottom=False)
            ax.tick_params(axis="y", direction="out")
            plt.legend(loc='upper right')

            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            from matplotlib.patches import Rectangle
            # inset a second axis for eigenvectors
            horizontal_position = 0.03
            vertical_position = 0.03
            # axins = ax.inset_axes([horizontal_position, vertical_position, 0.7, 0.45])
            n_eigenvectors = 150
            axins = inset_axes(ax, loc='lower left', width=f'{45*n_eigenvectors/molecule_size/1.5}%', height='55%', borderpad=0.15)
            mat = Usmall[:1*row_max, :n_eigenvectors]        # first 63 eigenvectors
            vmax = np.abs(mat).max()
            axins.imshow(mat, cmap="seismic", vmax=vmax, vmin=-vmax)
            axins.tick_params(left=False,
                           bottom=False,
                           labelleft=False,
                           labelbottom=False)
            axins.set_title('eigenvectors')

            c = '#CC78BC'
            # plot some arrows
            ax.annotate('', xy=(-18, 1e-4), xytext=(n_eigenvectors, 1e-4), annotation_clip=False,
                        arrowprops=dict(arrowstyle="< - >", color=c, linewidth=1))
            # ax.annotate('', xy=(30, 5e-7), xytext=(1180, 5e-7), annotation_clip=False,
            #             arrowprops=dict(arrowstyle="- | >", color='black', linewidth=1))
            # ax.annotate('eigenvectors', xy=(84, 1e-3), xytext=(300, 1e-7), annotation_clip=False,
            #             arrowprops=dict(arrowstyle="< -", color='black', linewidth=1), fontsize=8)
            ax.set_xlim([0, len(s)])

            # highlight zooming region
            # rect = Rectangle((0, 4e-5), n_eigenvectors, 22, fill=False, alpha=0.5, edgecolor=c, facecolor=c,
            #                  lw=3)
            # ax.add_patch(rect)

            for key_position in axins.spines:       # ordered dictionary of all four b
                axins.spines[key_position].set(alpha=0.5, linewidth=3, color=c)

            plt.tight_layout(pad=0.1)

            plt.figure(f'eigenvectors_{task["dataset_name"]}', figsize=(10., 3.2))
            mat = Usmall[:2 * row_max, :400]  # first 63 eigenvectors
            vmax = np.abs(mat).max()
            plt.imshow(mat, cmap="seismic", vmax=vmax, vmin=-vmax)
            ax = plt.gca()
            ax.tick_params(left=False,
                              bottom=False,
                              labelleft=False,
                              labelbottom=False)
            plt.tight_layout(pad=0.1)

            K_ridge = K_small + lam * np.eye(n)

            _, s_ridge, _ = sp.linalg.svd(K_ridge)
            _, s, _ = sp.linalg.svd(K_small)

            k = 400
            percentage_precon = float(k / n)
            inv_precon_operator = svd_preconditioner(U_eigenvectors=Usmall, s_eigenvalues=ssmall,
                                                     k_rank=int(percentage_precon * n))
            K_small_precon = inv_precon_operator @ (K_ridge)
            Usmall_precon, ssmall_precon, Vsmall_precon = sp.linalg.svd(K_small_precon)

            plt.figure(f'eigenvalues_{task["dataset_name"]}', figsize=(8, 1.6))
            plt.plot(s/s_ridge.min(), '--', c='#CC0000')
            # plt.plot(ssmall_precon, label=f'Preconditioned', c='#006600')
            plt.plot(s_ridge/s_ridge.min(), label='Original', c='#CC0000')
            plt.plot(np.roll(ssmall_precon, k), '-.', label=f'Preconditioned', c='#006600')




            plt.semilogy()
            # plt.legend(title='Eigenvalues', title_fontsize='x-small')
            # ax = plt.gca()
            # ax.tick_params(left=True,
            #                bottom=True,
            #                labelleft=False,
            #                labelbottom=False)
            plt.tight_layout(pad=0.1)


            fig = plt.figure('paper')
            ax = plt.subplot(2, 1, 1)

            mat = Usmall[:row_max, :col_max]
            vmax = np.abs(mat).max()
            plt.imshow(mat, cmap="seismic", vmax=vmax, vmin=-vmax)
            ax.tick_params(left=False,
                           bottom=False,
                           labelleft=False,
                           labelbottom=False)
            ax.grid(False)

            ax = plt.subplot(2, 1, 2)
            ax.plot(s, label='eigenvalues')
            # ax.plot(ssmall, label='eigenvalues')
            plt.plot(np.abs(U.T @ y), label='kernel PCA coefficients', linewidth=0.5, alpha=0.5)
            ax.semilogy()
            ax.tick_params(left=True,
                           bottom=False,
                           labelleft=True,
                           labelbottom=False)
            ax.set_xlim([0, len(s)])
            plt.legend()
            plt.tight_layout(pad=0.1)
            # ==============================================================================================================
            # LEVERAGE SCORES
            plt.figure('lev_scores')
            index_null_space = (s < 100 * s.min()).argmax()
            list_ridge_lam = []
            for k in [100, 200, int(0.4*index_null_space), int(0.7*index_null_space), index_null_space, n]:
                s_rank = np.zeros(n)
                s_rank[:k] = s[:k]
                K_rank_k = U @ np.diag(s_rank) @ V
                lam_ridge = np.linalg.norm(K - K_rank_k, ord='fro') ** 2 / k
                list_ridge_lam.append(lam_ridge)
                lev_scores = np.linalg.norm(U[:, :k], axis=1)**2
                plt.plot(lev_scores, label=k)
            plt.legend()
            plt.xlabel('columns')
            plt.ylabel('leverage scores')
            plt.semilogy()
            plt.tight_layout(pad=0.1)

            k = index_null_space
            lev_scores200 = np.linalg.norm(U[:, :k], axis=1)**2
            lev_scores_sorted_matrix = np.zeros((molecule_size, n_train))
            for i in range(n_train):
                lev_scores_sorted_matrix[:, i] = lev_scores200[i*molecule_size:(i+1)*molecule_size]



            # RIDGE leverage scores
            plt.figure('ridge leverage scores')
            ax1 = plt.subplot(2, 1, 1)


            # for lam in np.geomspace(1e-6, 1e-13, 5):
            for lam_temp in [1e-5, 1e-8, 1e-10, 1e-12]:
                K_lam = K.copy() @ K.T.copy() + lam_temp * np.eye(n)
                temp = sp.linalg.solve(K_lam, K)
                tau = np.diag(K.T @ temp)
                # U, s, V = sp.linalg.svd(K_lam)
                # ridge_lev_scores = np.linalg.norm(U, axis=1)**2

                # ax1.plot(tau, label=f'ridge: {lam:.0e}')
                ax1.plot(np.sort(tau), label=f'{lam:.0e}')
            ax1.legend(ncol=4, handlelength=1, columnspacing=1)
            ax1.set_ylim([-0.1, 1.1])
            # ax1.semilogy()

            ax2 = plt.subplot(2, 1, 2)

            # calculate ridge leverage scores
            lam = 1e-10
            K_lam = K.copy() @ K.T.copy() + lam * np.eye(n)
            temp = sp.linalg.solve(K_lam, K)
            tau = np.diag(K.T @ temp)

            # sort leverage scores periodically wrt to training points
            lev_scores_sorted_matrix = np.zeros((molecule_size, n_train))
            for i in range(n_train):
                lev_scores_sorted_matrix[:, i] = tau[i*molecule_size:(i+1)*molecule_size]

            mat = lev_scores_sorted_matrix[:, :20]
            vmax = np.abs(mat).max()
            cbar = ax2.imshow(mat.T, cmap="YlOrRd", vmax=vmax, vmin=0)
            ax2.set_ylabel('data')
            ax2.set_xlabel('atomic dimension')
            ax2.tick_params(left=False,
                            bottom=False,
                            labelleft=False,
                            labelbottom=False)
            plt.colorbar(cbar)

            plt.tight_layout(pad=0.1)

            plt.figure(f'lev_score matrix: {k}')
            mat = lev_scores_sorted_matrix
            vmax = np.abs(mat).max()
            plt.imshow(mat, cmap="YlOrRd", vmax=vmax, vmin=0)
            plt.xlabel('different training points')
            plt.ylabel('different atoms')
            plt.colorbar()
            plt.tight_layout()

        # plt.figure(figsize=(7, 4))
        # _, s, _ = scipy.linalg.svd(K)
        # plt.plot(s, label='orginal')
        # plt.semilogy()
        #
        # for p in [0.3, 0.6, 0.7, 0.8, 0.9]:
        #     mask = np.zeros(n, dtype=np.bool)
        #     mask[:int(p*n)] = 1
        #     np.random.shuffle(mask)
        #     _, s, _ = scipy.linalg.svd(K[mask])
        #     plt.plot(s, label=p)
        # plt.legend()
        # plt.tight_layout()
        return lin_inv_operator, k_final
