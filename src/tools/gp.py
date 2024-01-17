import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.gaussian_process as gp
import cholesky


def get_1d_data(n_train: int, n_test: int) -> [np.ndarray, np.ndarray]:
    x_train = -2 + np.random.random_sample(n_train).reshape(-1, 1) * 8
    x_test = np.linspace(-3, 8, n_test).reshape(-1, 1)
    return x_train, x_test


def get_2d_data(n_train: int, n_test: int) -> [np.ndarray, np.ndarray]:
    x_train = np.random.random_sample((n_train, 2)) * 10. - 5.

    x_points = np.linspace(-4.5, 4.5, n_test)
    x1, x2 = np.meshgrid(x_points, x_points)
    x_test = np.stack([x1.flatten(), x2.flatten()]).T
    return x_train, x_test


def f(x: np.ndarray, period=2.5):
    assert x.ndim == 2
    y = 2 * np.sin(2 * np.pi * x[:, 0] / period) * (1 + x[:, 0] / 5) ** 2
    if x.shape[1] == 2:
        x1 = x[:, 0]
        x2 = x[:, 1]
        # y = x1**2 + x2**2       # sphere function
        y = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2        # Himmelblaus's function
    return y


def gaussian_process(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, cov_function, noise_level=1E-8):
    K_tt = cov_function(x_train) + noise_level * np.eye(x_train.shape[0])
    K_ts = cov_function(x_train, x_test)
    K_ss = cov_function(x_test)

    # use custom cholesky to invert K_tt
    L, _ = cholesky.cholesky_decompostition(K_tt)
    # alpha = scipy.linalg.cho_solve((L, True), y_train)  # solves L.T @ (L @ x) = y_train

    # use CG
    alpha, info = scipy.sparse.linalg.cg(K_tt, y_train)

    y_prediction = K_ts.T.dot(alpha)

    factor = scipy.linalg.cho_solve((L, True), K_ts)
    cov_prediction = K_ss - K_ts.T.dot(factor)

    likelihood = - y_train.dot(alpha) - np.log(np.diag(L)).sum() - L.shape[0]/2*np.log(2*np.pi)
    return y_prediction, cov_prediction, likelihood


if __name__ == '__main__':

    n_points = 30
    eps = 1E-8
    n_dim = 2

    kernel = gp.kernels.RBF(1)
    k_fct = kernel.__call__

    if n_dim == 1:
        x_train, x_test = get_1d_data(n_train=10, n_test=100)
        y_train = f(x_train)

        mean_s, Sigma_star, _ = gaussian_process(x_train, y_train, x_test, k_fct)

        # L = scipy.linalg.cholesky(Sigma_star + 1E-5 * np.eye(Sigma_star.shape[0])).T
        # for i in range(3):
        #     unit_normal_points = np.random.randn(n_test).reshape(-1, 1)
        #     f_star = mean_s + L.dot(unit_normal_points)
        #     plt.plot(x_s, f_star, '--')
        plt.plot(x_test, f(x_test), 'gray', label='True', alpha=0.6)
        plt.scatter(x_train, y_train)
        plt.plot(x_test, mean_s, label='mean', linewidth=2.5)
        std = np.sqrt(np.diag(Sigma_star))
        mean = np.squeeze(mean_s)
        plt.fill_between(np.squeeze(x_test), mean-2*std, mean+2*std, alpha=0.5)
        plt.legend()

    # 2D - data
    if n_dim == 2:
        n_test = 20
        n_train = 1500
        shape_grid = [n_test, n_test]
        x_train, x_test = get_2d_data(n_train=n_train, n_test=n_test)
        y_train = f(x_train)

        mean_s, Sigma_star, _ = gaussian_process(x_train, y_train, x_test, k_fct)

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_test[:, 0].reshape(shape_grid), x_test[:, 1].reshape(shape_grid), f(x_test).reshape(shape_grid),
                        alpha=0.5, color='gray')

        vmax = np.abs(mean_s).max()
        ax.plot_surface(x_test[:, 0].reshape(shape_grid), x_test[:, 1].reshape(shape_grid), mean_s.reshape(shape_grid),
                        alpha=0.9, cmap='bwr', vmax=vmax, vmin=-vmax)

        ax.scatter(x_train[:, 0], x_train[:, 1], y_train, color='blue')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
