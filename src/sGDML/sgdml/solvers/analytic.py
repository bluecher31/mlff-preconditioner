#!/usr/bin/python

# MIT License
#
# Copyright (c) 2020 Stefan Chmiela
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

import sys
import logging
import warnings
from functools import partial

import numpy as np
import scipy as sp
import timeit

from .. import DONE, NOT_DONE


class Analytic(object):
    def __init__(self, gdml_train, desc, callback=None):

        self.log = logging.getLogger(__name__)

        self.gdml_train = gdml_train
        self.desc = desc

        self.callback = callback

    def solve(self, task, R_desc, R_d_desc, tril_perms_lin, y):

        sig = task['sig']
        lam = task['lam']
        use_E_cstr = task['use_E_cstr']

        n_train, dim_d = R_d_desc.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        dim_i = 3 * n_atoms

        # Compress kernel based on symmetries
        col_idxs = np.s_[:]
        if 'cprsn_keep_atoms_idxs' in task:

            cprsn_keep_idxs = task['cprsn_keep_atoms_idxs']
            cprsn_keep_idxs_lin = (
                np.arange(dim_i).reshape(n_atoms, -1)[cprsn_keep_idxs, :].ravel()
            )

            # if cprsn_callback is not None:
            #    cprsn_callback(n_atoms, cprsn_keep_idxs.shape[0])

            # if solver != 'analytic':
            #    raise ValueError(
            #        'Iterative solvers and compression are mutually exclusive options for now.'
            #    )

            col_idxs = (
                    cprsn_keep_idxs_lin[:, None] + np.arange(n_train) * dim_i
            ).T.ravel()

        if self.callback is not None:
            self.callback = partial(
                self.callback,
                disp_str='Assembling kernel matrix',
            )

        K = self.gdml_train._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            self.desc,
            use_E_cstr=use_E_cstr,
            col_idxs=col_idxs,
            callback=self.callback,
        )

        # DEBUG code
        # low-rank model. replace K with an low-rank svd approximation
        # k = int(0.5*K.shape[0])
        # print(f'Use low-rank approximation: rank={k} with n={K.shape[0]}')
        # U, s, V = sp.linalg.svd(K)  # -np.diag(lam*np.ones(K.shape[0]))
        # s_save = s.copy()
        # s[k:] = s_save.min()
        # K_lowrank = U @ np.diag(s) @ V
        # K = K_lowrank
        # print('finished low-rank')

        # build a sparse model
        # molecule_size = self.desc.dim_i
        # n_train, n_atoms, d_space = task['F_train'].shape
        # n = task['F_train'].size  # matrix size K
        # abs_maximum = np.abs(K).max()
        # eps = 1 * abs_maximum
        # mask = np.abs(K) < eps
        # mask_molecule = np.zeros((molecule_size, molecule_size), dtype=np.bool)
        # for i_atom in range(n_atoms):  # defines a block-diagonal matrix with 3x3 blocks
        #     mask_molecule[3 * i_atom:3 * (i_atom + 1), 3 * i_atom:3 * (i_atom + 1)] = True
        #
        # # broadcast mask_molecule to K.shape
        # mask_temp = np.concatenate([mask_molecule for _ in range(n_train)])
        # mask_trainingpoints = np.concatenate([mask_temp for _ in range(n_train)], axis=1)
        # mask[mask_trainingpoints] = False  # do not allow to delete self-interaction between a single atom
        #
        # mask_oddparity = mask != mask.T
        # assert mask_oddparity.sum() == 0, 'only allow symmetric deletes'
        # print(
        #     f'percentage of deleted entries: {mask.sum() / mask.size} ==================================================')
        # K[mask] = 0
        # END DEBUG

        start = timeit.default_timer()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            if K.shape[0] == K.shape[1]:

                K[np.diag_indices_from(K)] -= 10**-10   # lam  # regularize

                if self.callback is not None:
                    self.callback = partial(
                        self.callback,
                        disp_str='Solving linear system (Cholesky factorization)',
                    )
                    self.callback(NOT_DONE)

                try:

                    # Cholesky
                    L, lower = sp.linalg.cho_factor(
                        -K, overwrite_a=True, check_finite=False
                    )
                    alphas = -sp.linalg.cho_solve(
                        (L, lower), y, overwrite_b=True, check_finite=False
                    )
                except np.linalg.LinAlgError:  # try a solver that makes less assumptions

                    if self.callback is not None:
                        self.callback = partial(
                            self.callback,
                            disp_str='Solving linear system (LU factorization)      ',  # Keep whitespaces!
                        )
                        self.callback(NOT_DONE)

                    try:
                        # LU
                        alphas = sp.linalg.solve(
                            K, y, overwrite_a=True, overwrite_b=True, check_finite=False
                        )
                    except MemoryError:
                        self.log.critical(
                            'Not enough memory to train this system using a closed form solver.\n'
                            + 'Please reduce the size of the training set or consider one of the approximate solver options.'
                        )
                        print()
                        sys.exit()

                except MemoryError:
                    self.log.critical(
                        'Not enough memory to train this system using a closed form solver.\n'
                        + 'Please reduce the size of the training set or consider one of the approximate solver options.'
                    )
                    print()
                    sys.exit()
            else:

                if self.callback is not None:
                    self.callback = partial(
                        self.callback,
                        disp_str='Solving overdetermined linear system (least squares approximation)',
                    )
                    self.callback(NOT_DONE)

                # least squares for non-square K
                alphas = np.linalg.lstsq(K, y, rcond=-1)[0]

        stop = timeit.default_timer()

        if self.callback is not None:
            dur_s = (stop - start) / 2
            sec_disp_str = 'took {:.1f} s'.format(dur_s) if dur_s >= 0.1 else ''
            self.callback(
                DONE,
                disp_str='Training on {:,} points'.format(n_train),
                sec_disp_str=sec_disp_str,
            )
        if self.gdml_train.return_K is True:
            return alphas, K
        else:
            return alphas