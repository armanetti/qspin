"""
Data loaders and simple data utilities for questionnaire datasets.

The loaders assume the file layout used in the original study (folders such as
``PWE_data``, ``MACH_data``, ``GCBS``, ...) under a common ``pathdata`` root.
They drop missing entries and apply a per-dataset key-selection rule before
returning the numeric matrix.
"""

import numpy as np
import pandas as pd


def load_data(dataset, pathdata):
    """Load and clean one of the supported questionnaire datasets.

    Parameters
    ----------
    dataset : str
        Dataset short name.  Supported values:
        ``pwe``, ``mach``, ``gcbs``, ``sd3``, ``rwas``, ``big5``, ``dass``,
        ``empathy``, ``acme``, ``iri``, ``ei``, ``hexaco``, ``cfcs``, ``hsns``,
        ``msscq``.
    pathdata : str
        Root directory under which the dataset folders live.

    Returns
    -------
    X : ndarray
        Numeric response matrix.  ``X[i, j]`` is the answer of subject ``i``
        to item ``j``.
    """
    if dataset == 'pwe':
        path = pathdata + '/PWE_data/data.csv'
        data = pd.read_csv(path, sep="\t").dropna()
        subkeys = [el for el in data.keys() if el[0] == 'Q' and el[-1] == 'A']
        for key in subkeys:
            data = data.loc[data[key] != 0]
        X = np.array(data[subkeys])

    elif dataset == 'mach':
        path = pathdata + '/MACH_data/data.csv'
        data = pd.read_csv(path, sep="\t").dropna()
        subkeys = [el for el in data.keys() if el[0] == 'Q' and el[-1] == 'A']
        X = np.array(data[subkeys])

    elif dataset == 'gcbs':
        path = pathdata + '/GCBS/data.csv'
        data = pd.read_csv(path).dropna()
        mykeys = data.keys()[:15]
        for key in mykeys:
            data = data.loc[data[key] != 0]
        X = np.array(data[mykeys])

    elif dataset == 'sd3':
        path = pathdata + '/SD3/data.csv'
        data = pd.read_csv(path, sep="\t").dropna()
        mykeys = data.keys()[:-2]
        for key in mykeys:
            data = data.loc[data[key] != 0]
        X = np.array(data[mykeys])

    elif dataset == 'rwas':
        path = pathdata + '/RWAS/data.csv'
        data = pd.read_csv(path, low_memory=False)
        RWAS_responses = [f'Q{i}' for i in range(1, 23)]
        for key in RWAS_responses:
            data = data.loc[data[key] != 0]
        X = np.array(data[RWAS_responses])

    elif dataset == 'big5':
        path = pathdata + '/IPIP-FFM-data-8Nov2018/data-final.csv'
        data = pd.read_csv(path, sep="\t").dropna()
        mykeys = list(data.keys())[:50]
        for key in mykeys:
            data = data.loc[data[key] != 0]
        X = np.array(data[mykeys], dtype=int)

    elif dataset == 'dass':
        path = pathdata + '/DASS_dataset/'
        dass_data = pd.read_excel(path + "dass42_dataset.xlsx", engine='openpyxl').dropna()
        dass_data = dass_data.drop(columns=['anxiety_score', 'depression_score', 'stress_score'])
        # gender originally stored in two languages; encode as 0/1
        dass_data['gender'] = dass_data['gender'].map({'Donna': 0, 'Uomo': 1})
        dass_data = dass_data.drop(columns=['age', 'gender'])
        X = np.copy(dass_data)

    elif dataset == 'empathy':
        acme_data = pd.read_csv(pathdata + '/empathy_datasets/acme_data.csv').dropna()
        iri_data = pd.read_csv(pathdata + '/empathy_datasets/iri_data.csv').dropna()
        ei_data = pd.read_csv(pathdata + '/empathy_datasets/ei_data.csv').dropna()
        acme_data['Q_gender'] = acme_data['Q_gender'].map({'Donna': 0, 'Uomo': 1})
        X = np.concatenate((
            iri_data.iloc[:, 2:-4].values,
            acme_data.iloc[:, 2:-3].values,
            ei_data.iloc[:, 2:-2].values
        ), axis=1)

    elif dataset == 'acme':
        acme_data = pd.read_csv(pathdata + '/empathy_datasets/acme_data.csv').dropna()
        acme_data['Q_gender'] = acme_data['Q_gender'].map({'Donna': 0, 'Uomo': 1})
        X = acme_data.iloc[:, 2:-3].values

    elif dataset == 'iri':
        iri_data = pd.read_csv(pathdata + '/empathy_datasets/iri_data.csv').dropna()
        X = iri_data.iloc[:, 2:-4].values

    elif dataset == 'ei':
        ei_data = pd.read_csv(pathdata + '/empathy_datasets/ei_data.csv').dropna()
        X = ei_data.iloc[:, 2:-2].values

    elif dataset == 'hexaco':
        path = pathdata + '/HEXACO/data.csv'
        data = pd.read_csv(path, sep="\t").dropna()
        subkeys = [el for el in data.keys() if el not in ['V1', 'V2', 'elapse', 'country']]
        for key in subkeys:
            data = data.loc[data[key] != 0]
        X = np.array(data[subkeys])

    elif dataset == 'cfcs':
        path = pathdata + '/CFCS/data.csv'
        data = pd.read_csv(path, sep="\t").dropna()
        subkeys = [el for el in data.keys() if el[0] == 'Q']
        for key in subkeys:
            data = data.loc[(data[key] != 0) & (data[key] != -1)]
        X = np.array(data[subkeys])

    elif dataset == 'hsns':
        path = pathdata + '/HSNS+DD/data.csv'
        data = pd.read_csv(path, sep="\t").dropna()
        subkeys = [el for el in data.keys() if el[0] in ['H', 'D']]
        for key in subkeys:
            data = data.loc[(data[key] != 0) & (data[key] != -1)]
        X = np.array(data[subkeys])

    elif dataset == 'msscq':
        path = pathdata + '/MSSCQ/data.csv'
        data = pd.read_csv(path, sep="\t").dropna()
        subkeys = [el for el in data.keys() if el[0] == 'Q']
        for key in subkeys:
            data = data.loc[(data[key] != 0) & (data[key] != -1)]
        X = np.array(data[subkeys])

    else:
        raise ValueError(
            "Dataset not recognized. Please choose from 'pwe', 'mach', "
            "'gcbs', 'sd3', 'rwas', 'big5', 'dass', 'empathy', 'acme', "
            "'iri', 'ei', 'hexaco', 'cfcs', 'hsns' or 'msscq'.")

    return X


def discretize(x, R):
    """Round a scalar ``x`` to the closest integer in [0, R-1]."""
    y = round(x)
    y = np.min([R - 1, y])
    y = np.max([0, y])
    return y


def reshuffle(X):
    """Independently shuffle each column of ``X`` across rows.

    The result has the same per-column marginals as ``X`` but destroys all
    inter-column correlations -- useful as a null model for the dependence
    structure.
    """
    Y = np.copy(X)
    N, M = Y.shape
    ssi = np.random.choice(range(N), (N, M))
    return np.array([Y[ssi[:, i], i] for i in range(M)]).T


# ---------------------------------------------------------------------------
# Small plotting helpers — kept here because they manipulate the data view
# ---------------------------------------------------------------------------

def compare2matrices(matrixtoplot1, matrixtoplot2, mymin=None, mymax=None, filename=None):
    """Side-by-side matshow comparison of two matrices on a common colour scale."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colorbar import Colorbar

    dim = np.shape(matrixtoplot1)[0]
    fig, axes = plt.subplots(nrows=1, ncols=2)

    if mymin is None:
        mymin = min(np.min(matrixtoplot1), np.min(matrixtoplot2))
    if mymax is None:
        mymax = max(np.max(matrixtoplot1), np.max(matrixtoplot2))
    for ax, mat in zip(axes.flat, [matrixtoplot1, matrixtoplot2]):
        ax.matshow(mat, vmin=mymin, vmax=mymax)
        ax.xaxis.set_ticks(range(dim))
        ax.yaxis.set_ticks(range(dim))

    cbar_ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    norm = mcolors.Normalize(vmin=mymin, vmax=mymax)
    Colorbar(ax=cbar_ax, norm=norm)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot2matrices(mat1, mat2, ticksspacing=None, filename=None,
                  removediagonal=True, xlabel=None, ylabel=None, title=None):
    """Plot two matrices superimposed across the diagonal of one matshow."""
    import matplotlib.pyplot as plt

    D = len(mat1)
    if ticksspacing is None:
        ticksspacing = max(1, int(D / 5))

    mat3 = np.copy(mat1)
    mask = [[i < j for i in range(D)] for j in range(D)]
    mat3[mask] = mat2[mask]
    if removediagonal:
        mat3 -= np.eye(D) * np.diag(mat3)
    plt.matshow(mat3)
    plt.colorbar()
    plt.xticks(range(0, D, ticksspacing))
    plt.yticks(range(0, D, ticksspacing))
    if title is not None:
        plt.title(title, fontsize=20)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=20)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=20)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def remove_diagonal(M):
    """Return ``M`` with its diagonal zeroed out."""
    return M - np.diag(M) * np.eye(len(M))
