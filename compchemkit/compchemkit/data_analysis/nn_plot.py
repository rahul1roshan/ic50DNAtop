import numpy as np
from scipy import sparse
from compchemkit.machine_learning.kernel import tanimoto_from_sparse
import seaborn as sns
import matplotlib.pyplot as plt


def nn_sim(source_fp: sparse.csr_matrix, target_fp: sparse.csr_matrix, ignore_diag=False, batch_size=1000, add_index=False):
    row_max = []
    idx = []
    if ignore_diag and source_fp.shape != target_fp.shape:
        raise IndexError("Matrices must be of same shape to determine diagonal")
    for i in range(0, source_fp.shape[0], batch_size):
        batch = source_fp[i:i + batch_size]
        batch_sim = tanimoto_from_sparse(batch, target_fp)
        if ignore_diag:
            end_row = min(source_fp.shape[0], i+batch_size)
            diag_idx = (np.arange(0, batch_sim.shape[0]), np.arange(i, end_row))
            batch_sim[diag_idx] = np.nan
        row_max.append(np.nanmax(batch_sim, axis=1))
        if add_index:
            idx.append(np.nanargmax(batch_sim, axis=1))
    if add_index:
        return np.hstack(row_max), np.hstack(idx)
    return np.hstack(row_max)


def nn_plot(x_similarity, y_similarity, color="blue", scatter=False, scatter_kwargs=None):
    if scatter_kwargs is None:
        scatter_kwargs = dict()

    color_dict = {"blue": {"face_color": np.array([0.96862745098039216, 0.98431372549019602, 1]),
                           "plot_color": "Blues",
                           "line_color": "tab:blue",
                           },
                  "red": {"face_color": np.array([1, 0.97431372549019602, 0.96862745098039216]),
                          "plot_color": "Reds",
                          "line_color": "tab:red",
                          },
                  }
    color_dict = color_dict[color]
    out = sns.jointplot(x=x_similarity, y=y_similarity, kind="kde", cmap=color_dict["plot_color"],
                        color=color_dict["line_color"], joint_kws={"clip": (0, 1), "thresh": 0.05}, shade=True)
    out.fig.set_size_inches(6, 6)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0, hspace=0)
    ax = out.fig.get_axes()[0]
    ax.set_facecolor(color_dict["face_color"])
    ax.plot((0, 1), (0, 1), c="k", ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1, 0])
    if scatter:
        scatter_default_kwargs = {"marker": "x",
                                  "color": "grey",
                                  "s": 10,
                                  "alpha": 0.2
                                  }
        scatter_default_kwargs.update(scatter_kwargs)

        ax.scatter(x_similarity, y_similarity, **scatter_default_kwargs)
    return out, ax
