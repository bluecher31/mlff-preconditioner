from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from geometric import coordinates_to_distance_matrix
import matplotlib.pyplot as plt
import itertools as it

from project.tools.geometric import coordinates_to_distance_matrix
from project.tools.plot_data import map_dataset_name_to_molecule


def atomic_color(x):
    if x == 1:
        return "darkgrey"
    elif x == 6:
        return "black"
    elif x == 7:
        return "darkgreen"
    elif x == 8:
        return "darkred"
    else:
        print('No color specified for atomic type {}. Setting it to black.'.format(x))
        return "k"


# TODO: in the future we should replace bond matrix by some function which returns a coefficients for entry i,j as we
#  can then use the routine below for all kinds of pairwise interaction plots
def plot_marked_atom(R, z, z_ind, bond_matrix=None, elev=0, azim=0, title: str = ''):
    """
    Plot the molecule with one (multiple) atom(s) marked.
    Args:
        R (array_like): The atomic positions of shape [n,3]
        z (array_like): The atomic types of shape [n]
        z_ind (int or array_like): index or multiple indices which atoms shall be marked
        bond_matrix(array_like): pairwise bond matrix which has one for bond and zero otherwise
        elev (float): elevation angle that controls the view of the plot
        azim (float): azimuthal angle that controls the view of the plot

    Returns: axis

    """
    mask = np.zeros(R.shape[0], dtype=np.bool)
    # mask[z_ind] = True

    fig = plt.figure(title, figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    for n, (z_, r, msk) in enumerate(zip(z, R, mask)):
        marker = lambda x: ("lightgreen", 3) if x else (None, 0)
        ax.scatter(r[0], r[1], r[2],
                   c=atomic_color(z_),
                   edgecolors=marker(msk)[0],
                   linewidths=marker(msk)[1],
                   s=400)
        if bond_matrix is not None:
            for i in range(R.shape[0]):
                if i < n:
                    r_ = R[i, ...]
                    a = bond_matrix[n, i]
                    if a != 0:
                        ax.plot([r[0], r_[0]], [r[1], r_[1]], [r[2], r_[2]], lw=1, c="k", ls='--')
                        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.axis(False)
    plt.tight_layout(pad=0.1)
    return ax


def plot_atomic_contributions(R, z, contributions, bond_matrix=None, elev=0, azim=0, title: str = '',
                              marker_size: float = 1.):
    """
    Plot the molecule with one (multiple) atom(s) marked.
    Args:
        R (array_like): The atomic positions of shape [n,3]
        z (array_like): The atomic types of shape [n]
        z_ind (int or array_like): index or multiple indices which atoms shall be marked
        bond_matrix(array_like): pairwise bond matrix which has one for bond and zero otherwise
        elev (float): elevation angle that controls the view of the plot
        azim (float): azimuthal angle that controls the view of the plot

    Returns: axis

    """
    # Ideas to visualize atomic contributions
    #   - rescale size of atoms
    #   - color encode contribution
    #   - use transparency (alpha parameter)
    assert contributions.shape == z.shape
    assert len(R) == len(z)
    mask = np.zeros(R.shape[0], dtype=np.bool)
    # mask[z_ind] = True

    fig = plt.figure(title, figsize=(5, 5), facecolor='green')
    ax = fig.add_subplot(111, projection='3d')
    for n, (z_, r, msk, ctr) in enumerate(zip(z, R, mask, contributions)):
        marker = lambda x: ("lightgreen", 3) if x else (None, 0)
        ax.scatter(r[0], r[1], r[2],
                   c=atomic_color(z_),
                   edgecolors=marker(msk)[0],
                   linewidths=marker(msk)[1],
                   s=marker_size*400, alpha=ctr)
        if bond_matrix is not None:
            for i in range(R.shape[0]):
                if i < n:
                    r_ = R[i, ...]
                    a = bond_matrix[n, i]
                    if a != 0:
                        ax.plot([r[0], r_[0]], [r[1], r_[1]], [r[2], r_[2]], lw=1, c="k", ls='--')
                        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.axis(False)
    plt.tight_layout(pad=0.1)
    return ax


""" Usage example to plot aspirin: 

    import numpy as np
    import matplotlib.pyplot as plt
    from geometric_attention.utils.geometric import coordinates_to_distance_matrix

    test_ds = np.load("../data/MD17/aspirin.npz")

    R = test_ds['R'][20000]
    # R.shape = [n,3]
    z = test_ds['z']
    # R.shape = [n]
    bond_matrix = (coordinates_to_distance_matrix(R) < 1.6).astype(np.int)

    plot_marked_atom(R, z, z_ind=0, bond_matrix=bond_matrix, elev=80, azim=10)
    plt.show()
"""


def regular_grid(shape):
    axes = np.empty(shape, dtype=object)
    for i, j in it.product(range(shape[0]), range(shape[1])):
        axes[i, j] = plt.subplot2grid(shape, (i, j))
    return axes


def plot_single_molecule(dataset_npz, index: int):
    name_dataset_bytes = dataset_npz['name'].item()
    name_dataset = map_dataset_name_to_molecule(name_dataset_bytes.decode())
    R = dataset_npz['R'][index]
    # R.shape = [n,3]
    z = dataset_npz['z']
    # R.shape = [n]
    bond_matrix = (coordinates_to_distance_matrix(R) < 1.6).astype(np.int)

    plot_marked_atom(R, z, z_ind=0, bond_matrix=bond_matrix, elev=80, azim=10, title=f'{name_dataset}_{index}')
    plt.show()


def plot_single_molecule_contributions(dataset_npz, index: int):
    name_dataset_bytes = dataset_npz['name'].item()
    name_dataset = map_dataset_name_to_molecule(name_dataset_bytes.decode())
    R = dataset_npz['R'][index]
    # R.shape = [n,3]
    z = dataset_npz['z']
    # R.shape = [n]
    bond_matrix = (coordinates_to_distance_matrix(R) < 1.6).astype(np.int)

    plot_marked_atom(R, z, z_ind=0, bond_matrix=bond_matrix, elev=80, azim=10, title=f'{name_dataset}_{index}')
    plt.show()

