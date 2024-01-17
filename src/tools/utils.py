import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
from scipy.linalg import inv
import scipy
from typing import List
import time
from tqdm import trange, tqdm

from sgdml import train
from . import gp as custom_gp
from . import cholesky as cholesky


def callback(*args, **kwargs):
    pass


def test_function(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2
    # Sqphere function
    f = np.sum(x**2, axis=1)
    return f


def jackknife(samples: np.ndarray):
    """Copyright: Julian Urban. Return mean and estimated lower error bound."""
    means = []

    for i in range(samples.shape[0]):
        means.append(np.delete(samples, i, axis=0).mean(axis=0))

    means = np.asarray(means)
    mean = means.mean(axis=0)
    error = np.sqrt((samples.shape[0] - 1) * np.mean(np.square(means - mean), axis=0))
    
    return mean, error


def test_cholesky(K, permute=True, print_log=False, break_percentage=1.):
    L, list_of_permutations = cholesky.cholesky_decompostition(np.copy(K), permute=permute,
                                                               break_percentage=break_percentage)

    K_hat = L.dot(L.T)
    if permute is True:
        K_hat = cholesky.pivot_transformation(K_hat, list_of_permutations)
        Kinv_hat = inv(K_hat)
        identity = Kinv_hat.dot(K)
    else:
        identity = scipy.linalg.cho_solve((L, True), K)

    accuracy = np.linalg.norm(K_hat - K)
    print('start calc condition number')
    condition_number = np.linalg.cond(identity)
    if print_log:
        print(f"#filled entries:                            {2*(1 - np.sum(L==0)/L.size)*100:.1f}%")
        print(f"np.abs(L).min():                            {np.abs(L[L>0]).min():.3E}")
        print(f"K all close L.dot(L.T):                     {np.allclose(K, K_hat)}")
        print(f"|| K - L.dot(L.T) || =                      {accuracy:.2E}")
        print(f'cond(Kinv_hat * K) =                        {condition_number:.2E}\n')
    return accuracy, condition_number


def solve_linear_system(A, b, pivoting=True, break_percentage=0.1):
    L, list_of_permutations = cholesky.cholesky_decompostition(A, break_percentage=break_percentage, permute=pivoting)

    if pivoting is True:
        A_pi = cholesky.pivot_transformation(A, list_of_permutations, forward=True)
        b_pi = cholesky.pivot_transformation(b, list_of_permutations,
                                             is_matrix=False, forward=True)  # forward transformation
    else:
        A_pi = A.copy()
        b_pi = b.copy()

    def apply_inv(a):
        return scipy.linalg.cho_solve((L, True), a)

    M = scipy.sparse.linalg.LinearOperator(shape=L.shape, matvec=apply_inv)
    start = time.time()
    x, info = scipy.sparse.linalg.cg(A_pi, b_pi, M=M)
    end = time.time()
    t = end-start
    # print(f't = {t:.5}s')

    if pivoting is True:
        x = cholesky.pivot_transformation(x, list_of_permutations, is_matrix=False)  # reverse transformation

    error = np.linalg.norm(A @ x - b)
    n = A.shape[0]
    if info is not 0:
        print(f'preconditioned solution\n'
              f'info = {info}')
        print(f'error = {error / n:.2E}\n')
    return t


def solve_linear_system_woodbury(A, b, pivoting=True, break_percentage=0.1):
    lam = 1E-10
    print(f'\nstart solve_linear_system_woodbury\n'
          f'K.shape = {A.shape}\n'
          f'Memory size of numpy array: {A.size * A.itemsize * 2 ** -20:.1f}MB or {A.size * A.itemsize * 2 ** -30:.1f}GB ')
    L_pi, list_of_permutations = cholesky.cholesky_decompostition(A, break_percentage=break_percentage, permute=pivoting)

    print('set-up preconditioning')
    A_hat = A + lam * np.eye(A.shape[0])
    k = len(list_of_permutations)
    if pivoting is True:
        L = cholesky.pivot_transformation(L_pi, list_of_permutations, forward=False, is_matrix=False)
    else:
        L = L_pi.copy()
    L = L[:, :k]        # N x k

    kernel = lam * np.eye(k) + (L.T @ L)
    L2, _ = cholesky.cholesky_decompostition(kernel, permute=False)       # k x k
    temp_mat = scipy.linalg.solve_triangular(L2, L.T, lower=True)   # k x N
    del L, A, L_pi

    def apply_inv(a):   # a.shape: N or N x N
        temp_vec = temp_mat.T @ (temp_mat @ a)
        # temp_vec = kernel_inversion @ a
        x = lam**-1 * (a - temp_vec)
        return x

    M = scipy.sparse.linalg.LinearOperator(shape=A_hat.shape, matvec=apply_inv)
    # start = time.time()

    identity = M.matmat(A_hat)
    cond_number = np.linalg.cond(identity)

    global num_iters
    num_iters = 0
    global pbar
    def _cg_status(_: np.ndarray):
        global num_iters
        num_iters += 1
        pbar.update(1)

    pbar = tqdm(total=A_hat.shape[0]*10)
    if k > 3:
        x, info = scipy.sparse.linalg.cg(A_hat, b, M=M, callback=_cg_status)
    else:
        print('direct CG')
        x, info = scipy.sparse.linalg.cg(A_hat, b, callback=_cg_status)
    pbar.close()
    print(f'\n'
          f'num_iters = {num_iters}\n'
          f'K.shape = {A_hat.shape}\n'
          f'finished CG\n')

    error = np.linalg.norm(A_hat @ x - b)
    n = A_hat.shape[0]
    if info is not 0:
        print(f'\n======================\n'
              f'CG failed, k = {k}\n'
              f'=======================\n'
              f'info = {info}')
    print(f'error = {error / n:.2E}')
    return num_iters, cond_number


def get_sGDML_kernel_mat(n_train: int):
    from sGDML.sgdml.train import GDMLTrain
    # dataset = np.load('./sGDML/ethanol_dft.npz')
    dataset = np.load('./sGDML/aspirin_dft.npz')
    gdml_train = GDMLTrain(return_K=True)
    task = gdml_train.create_task(dataset, n_train,
                                  valid_dataset=dataset, n_valid=1000,
                                  sig=10, lam=1e-15, solver='analytic')
    model, K, y = gdml_train.train(task=task, callback=callback)
    return -K, -y


def create_kernel_mat(n=300, dim=2):
    assert dim == 1 or dim == 2
    if dim == 1:
        width = 10
        rbf_kernel = gp.kernels.RBF
        matern_kernel = gp.kernels.Matern
        x = np.linspace(-width/2, width/2, n).reshape(-1, 1)
        # K = kernel(Xtest, Xtest, 1)
        K = rbf_kernel().__call__(x)
    elif dim == 2:
        x = np.random.random_sample((n, 2)) * 1.
        kernel = gp.kernels.RBF()
        K = kernel.__call__(x)
    K += 1E-10 * np.eye(K.shape[0])
    return K, test_function(x)


def random_kernel_mat(n=300):
    A = np.random.randn(n, n)
    K = A.dot(A.T)
    return K


def two_dim_kernel_matrix(n=300):
    x, _ = custom_gp.get_2d_data(n_train=n, n_test=1)
    kernel = gp.kernels.RBF()
    K = kernel.__call__(x)
    return K


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

def print_information(K: np.ndarray):
    # eigvalgs = np.linalg.eigvals(K)
    # print(f'cond(K) =                                        {np.linalg.cond(K):.2E}')
    # print(f'smallest eigvals =                               {np.abs(eigvalgs).min():.2E}           sqrt()={np.sqrt(np.abs(eigvalgs).min()):.2E}')
    # print(f'biggest eigvals =                                {np.abs(eigvalgs).max():.2E}')
    print(f'smallest abs entry =                             {np.abs(K).min():.2E}\n')
    print(f'K.shape =                                        {K.shape}')


def plot_two_lists(x, y1, y2, xlabel, label1, label2, title=''):
    fig, ax1 = plt.subplots()

    ax1.set_title(title)

    color = 'red'
    # ax1.semilogx()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(label1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(x, y1, label='', color=color, marker='+', linestyle='--')

    ax2 = ax1.twinx()

    color = 'blue'
    ax2.semilogy()
    ax2.set_ylabel(label2, color=color)
    ax2.plot(x, y2, label='', color=color, marker='x', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()


def plot_n_zeros(K: np.ndarray, n_zero_entries: np.ndarray, threshold: float, percentage: float):
    percentage_of_iterations = np.arange(K.shape[0]) * 100 / K.shape[0]
    percentage_of_iterations = percentage_of_iterations[:len(n_zero_entries)]
    normalization = K.shape[0] ** 2 / 100

    plt.figure(f"Sparsity for threshold = {threshold:.3E}")
    plt.plot(percentage_of_iterations, n_zero_entries / normalization, label='nominal')
    normalization = ((K.shape[0] - np.arange(K.shape[0])) ** 2 / 100)[:len(n_zero_entries)]
    plt.plot(percentage_of_iterations, n_zero_entries / normalization, label='rescaled')
    plt.xlabel('# percentage of iterations')
    plt.ylabel('% entries set to zero')
    plt.legend()


def plot_condition_number(percentage_list: List[float], condition_number: List[float]):
    plt.figure('Condition number vs early stop')
    if isinstance(percentage_list, list):
        percentage_list = np.array(percentage_list)
    plt.plot(np.round(percentage_list*100, 1), condition_number, 'o')
    plt.semilogy()
    plt.xlabel('early stop in %')
    plt.ylabel('condition number')


def visualize_mat(arr: np.ndarray, name: str = None):
    assert arr.ndim == 2
    if name is None:
        plt.figure()
    else:
        plt.figure(name)
    vmax = np.abs(arr).max()
    plt.imshow(arr, cmap='bwr', vmax=vmax, vmin=-vmax)
    plt.colorbar()


def load_kernel_matrix(dataset_npz, n_datapoints: int) -> np.ndarray:
    gdml_train = train.GDMLTrain(use_torch=False, max_processes=1, return_K=True)
    task = gdml_train.create_task(dataset_npz, int(n_datapoints),
                                  valid_dataset=dataset_npz, n_valid=1000,
                                  sig=10, lam=1e-15, solver='analytic')

    def callback(*args, **kwargs):
        pass

    model, K, alphas = gdml_train.train(task=task, break_percentage=0.1, callback=callback,
                             str_preconditioner='random_scores2', flag_eigvals=False)
    return K
