import matplotlib.pyplot as plt
import numpy as np

import tools.utils as utils
# import tools.incomplete_cholesky as ichol
import sgdml.solvers.incomplete_cholesky as ichol

from project.tools import custom_cg_solver as custom_cg
from tools import plot_data
from tools import create_data


def main_sGDML():
    from sGDML.sgdml.train import GDMLTrain
    dataset = np.load('./sGDML/ethanol_ccsd_t-train.npz')
    # dataset = np.load('./sGDML/ethanol_dft.npz')

    gdml_train = GDMLTrain()
    task = gdml_train.create_task(dataset, 30,
                                  valid_dataset=dataset, n_valid=1000,
                                  sig=10, lam=1e-15, solver='analytic')
    model, K, y = gdml_train.train(task)

    np.savez_compressed('./sGDML/m_ethanol.npz', **model)
    # test in terminal
    #  sgdml test m_ethanol.npz ethanol_ccsd_t-test.npz 100


def main_test_cholesky():
    t = []
    cond_number = []
    n = 2
    percentage = np.linspace(0.05, 0.3, 2)

    for p in percentage:
        print(f'\npercentage = {p}')
        temp_iterations = np.zeros(n)
        temp_cond = np.zeros(n)
        for i in range(n):
            # K, y = utils.create_kernel_mat(n=1000, dim=2)
            K, y, _ = utils.get_sGDML_kernel_mat(n_train=30)
            temp_iterations[i], temp_cond[i] = utils.solve_linear_system_woodbury(K, y, break_percentage=p, pivoting=True)
        t.append(temp_iterations.mean())
        cond_number.append(temp_cond.mean())
        print(f'chol iteration = {p*K.shape[0]}')

    utils.plot_two_lists(percentage, t, cond_number, xlabel='break percentage', label1='CG steps', label2='cond_number',
                         title=f'K.shape = {K.shape}, n_repeat = {n}')


def main_incomplete_cholesky():
    K, y = utils.create_kernel_mat(n=5000, dim=2)

    print(f'K.shape = {K.shape}')
    k = int(0.05 * K.shape[0])

    get_col_K = lambda i: utils.get_col(K, i)
    L, index_columns = ichol.pivoted_cholesky(get_col_K, diagonal=np.diag(K), max_rank=k)
    print(f'remaining error = {np.linalg.norm(K - L @ L.T)}')


def main_plot():
    n_datapoints = 100
    list_percentage = [0.01, 0.02, 0.05, 0.1]
    # list_column = [250, 500, 750]

    eigvals = False
    cg_steps = True

    if eigvals is True:
        x, y, labels = create_data.eigvals(n_datapoints=n_datapoints, list_percentage=list_percentage)
        plot_data.effect_preconditioning(x_fraction_precon=x, y_observable=y, label_list=labels, title='eigvals',
                                         solid_line=True)

    if cg_steps is True:
        # x, y, labels = create_data.cg_steps(n_datapoints=n_datapoints, list_columns=list_column)
        x, y, labels = create_data.cg_steps(n_datapoints=n_datapoints, list_percentage=list_percentage)

        plot_data.effect_preconditioning(x_fraction_precon=x, y_observable=y, label_list=labels,
                                         title=f'#cg steps, aspirin, n={n_datapoints} train_pnts')


def main_analyse_eigvals():
    lam = 1E-9
    # list_percentage = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]     # small grid
    list_percentage = [0.01, 0.05, 0.1, 0.2, 0.5]

    K, y = utils.get_sGDML_kernel_mat(n_train=30)
    K_hat = K + lam * np.eye(K.shape[0])

    eigvals_K_hat = np.linalg.eigvals(K_hat)
    eig = np.abs(eigvals_K_hat)
    plt.figure('eigvals depending on preconditioning')
    plt.title(f'K.shape = {K_hat.shape}')
    plt.plot(eig, label='K_hat')
    plt.semilogy()

    eigvals_K = np.linalg.eigvals(K)
    eig = np.abs(eigvals_K)
    plt.plot(eig, label='K')

    for break_percentage in list_percentage:
        M = custom_cg.init_percond_operator(K, lam, break_percentage)
        eigvals_P_K = np.linalg.eigvals(M @ K_hat)
        eig = np.abs(eigvals_P_K)
        plt.plot(eig, label=break_percentage)
    plt.legend()



if __name__ == '__main__':

    # main_test_cholesky()
    # main_sGDML()
    # main_incomplete_cholesky()

    # main_analyse_eigvals()
    main_plot()

