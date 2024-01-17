import numpy as np
import matplotlib.pyplot as plt

from project.tools.utils import load_kernel_matrix, visualize_mat
from tools.create_data import get_dataset, get_number_of_atoms, normalize_to_aspirin
from tools.plot_routines_molecules import plot_single_molecule, plot_atomic_contributions
from project.tools.geometric import coordinates_to_distance_matrix

from pathlib import Path


def calculate_atomic_contributions(K: np.ndarray, n_datapoints: int, n: int, d: int) -> np.ndarray:
    assert K.shape == (n, n)
    assert n == 3 * n_datapoints * d
    # load kernel matrix
    U, s, _ = np.linalg.svd(K)
    # U: shape (n, n), (spatial-xyz, kernel-dimension), U[:, 10] = 10th eigenvector

    # visualize_mat(arr=U[:, :k], name='eigenvectors')
    U_sorted = np.reshape(U, (n_datapoints, 3 * d, n))     # shape: (3d, n_training. n)
    # visualize_mat(arr=U_sorted[index_molecule][:, :k], name=f'sorted eigenvector: {index_molecule}')
    average_eigenvector = U_sorted.mean(0)
    atomic_contributions = np.sqrt(np.mean(average_eigenvector.reshape(d, 3, -1)**2, axis=1))
    return atomic_contributions


def visualize_atomic_contributions(R: np.ndarray, z: np.ndarray, atomic_contributions: np.ndarray, i_eigvalue: int,
                                   index_molecule: int, save: bool = False):
    bond_matrix = (coordinates_to_distance_matrix(R) < 1.6).astype(np.int)
    contributions = atomic_contributions[:, i_eigvalue]
    contributions /= contributions.max()
    plot_atomic_contributions(R, z, contributions=contributions,
                              bond_matrix=bond_matrix, azim=74, elev=38, title=f'{name_dataset}_spectrum_{i_eigvalue}',
                              marker_size=2.5)
    plt.tight_layout(pad=-7)
    if save is True:
        fname = Path('.').absolute() / 'figures' / 'molecules' / 'eigenvectors' / name_dataset / \
                f'{name_dataset}_sample_{index_molecule}_spectrum_{i_eigvalue}.pdf'
        fname.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(fname=fname)
        plt.close()


def script_visualize_atoms(name_dataset: str):
    index_molecule = 10
    n_datapoints = 4

    test_ds = get_dataset(path_to_script='./', name_dataset=name_dataset)
    # plot_single_molecule(dataset_npz=test_ds, index=6)

    d = get_number_of_atoms(dataset_name=name_dataset)
    n = 3 * d * n_datapoints
    K = load_kernel_matrix(dataset_npz=test_ds, n_datapoints=n_datapoints)
    assert K.shape == (n, n), f'Expected kernel shape: {(n, n)} but got {K.shape}'

    atomic_contributions = calculate_atomic_contributions(K=K, n_datapoints=n_datapoints, n=n, d=d)
    for i_eigvalue in range(10):
        visualize_atomic_contributions(R=test_ds['R'][index_molecule], z=test_ds['z'],
                                       atomic_contributions=atomic_contributions, i_eigvalue=i_eigvalue,
                                       index_molecule=index_molecule)


def script_investigate_kernel_matrix(name_dataset: str):
    test_ds = get_dataset(path_to_script='./', name_dataset=name_dataset)

    plt.figure(f'Stability of spectrum - {name_dataset}')
    for n_datapoints in np.linspace(3, 75, 5, dtype=int):
        n_datapoints = normalize_to_aspirin(n_datapoints, name_dataset)
        K = load_kernel_matrix(dataset_npz=test_ds, n_datapoints=n_datapoints)
        n = K.shape[0]
        U, s, _ = np.linalg.svd(K)
        plt.plot(np.linspace(0, 1, n), s, label=n_datapoints)

    plt.title(name_dataset)
    plt.xlabel('Fraction of singular values')
    plt.ylabel('Singular values')
    plt.semilogy()
    plt.legend()
    plt.tight_layout(pad=0.1)


def create_random_kernel(n: int, null_space: int, name: str) -> np.ndarray:
    l = n - null_space
    l = n
    if name == 'gaussian entries':
        random_matrix = np.random.randn(n ** 2).reshape(n, n)
        K = 0.5*(random_matrix + random_matrix.T)
    elif name == 'random basis':
        R = np.random.randn(n * l).reshape(n, l) * 0.00007
        K = R @ R.T + 1E-10 * np.eye(n)
    else:
        raise ValueError(f'Not defined: {name}')
    return K


def script_random_kernel_matrix(name_dataset: str):
    n_datapoints = 20
    n_datapoints = normalize_to_aspirin(n_datapoints, name_dataset)
    d = get_number_of_atoms(dataset_name=name_dataset)
    n = 3 * d * n_datapoints
    null_space = 6 * n_datapoints
    test_ds = get_dataset(path_to_script='./', name_dataset=name_dataset)

    # K_random = create_random_kernel(n=n, null_space=null_space, name='gaussian entries')
    # plt.figure('singular values')
    # U, s, _ = np.linalg.svd(K_random)
    # plt.plot(s, label='random')
    #
    # K = load_kernel_matrix(dataset_npz=test_ds, n_datapoints=n_datapoints)
    # U, s, _ = np.linalg.svd(K)
    # plt.plot(s, label=name_dataset)
    #
    # plt.semilogy()
    # plt.legend()
    # plt.tight_layout(pad=0.1)

    which = 'random basis'
    eigvals_list = []
    for i in range(10):
        if which in ['random basis', 'gaussian entries']:
            K = create_random_kernel(n, null_space=null_space, name=which)
            exclude_eigvals = 0
        elif which == 'kernel':
            K = load_kernel_matrix(dataset_npz=test_ds, n_datapoints=n_datapoints+i)
            exclude_eigvals = 500
        eigvals_random = np.linalg.eigvals(K)
        eigvals_list.append(eigvals_random[exclude_eigvals:])

    eigvals = np.abs(np.concatenate(eigvals_list))
    # eigvals_kernel = np.linalg.eigvals(K)
    plt.figure(f'wigner semicircle: {which}, exclude eigvals {exclude_eigvals}')
    plt.hist(np.squeeze(eigvals), bins=50, density=True)
    plt.xlabel('magnitude eigenvalue')
    plt.ylabel('density')
    plt.tight_layout(pad=0.1)


if __name__ == '__main__':
    script_name = 'visualize_atoms'
    script_name = 'investigate_kernel_matrix'
    script_name = 'random_kernel_matrix'

    name_dataset = 'aspirin'

    if script_name == 'visualize_atoms':
        script_visualize_atoms(name_dataset)

    elif script_name == 'investigate_kernel_matrix':
        script_investigate_kernel_matrix(name_dataset)

    elif script_name == 'random_kernel_matrix':
        script_random_kernel_matrix(name_dataset)

    else:
        raise ValueError(f'script: {script_name} not defined.')

