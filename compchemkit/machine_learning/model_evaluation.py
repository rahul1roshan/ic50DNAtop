from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from compchemkit.machine_learning.classifier import TanimotoKNN
from sklearn.model_selection import GridSearchCV
import numpy as np
from typing import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from .data_storage import DataSet


def evaluate_model(model, training_set: DataSet, test_set: DataSet) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not hasattr(model, 'predict'):
        TypeError("Model has no predict function")

    y_pred_training = model.predict(training_set.feature_matrix)
    y_pred_test = model.predict(test_set.feature_matrix)

    is_knn = False
    if isinstance(model, KNeighborsClassifier) or isinstance(model, TanimotoKNN):
        is_knn = True
    if isinstance(model, GridSearchCV):
        if isinstance(model.estimator, KNeighborsClassifier) or isinstance(model.estimator, TanimotoKNN):
            is_knn = True

    if not is_knn:
        y_score_training = model.predict_proba(training_set.feature_matrix)[:, 1]
        y_score_test = model.predict_proba(test_set.feature_matrix)[:, 1]
    else:
        y_score_training = None
        y_score_test = None

    training_metrics = evaluate_classification(training_set.label, y_pred_training, y_score_training)
    training_metrics["data_set"] = "training"

    test_metrics = evaluate_classification(test_set.label, y_pred_test, y_score_test)
    test_metrics["data_set"] = "test"

    return training_metrics, test_metrics


def evaluate_dataset_classification(model, dataset: DataSet) -> pd.DataFrame:
    if not hasattr(model, 'predict'):
        TypeError("Model has no predict function")

    y_pred = model.predict(dataset.feature_matrix)

    is_knn = False
    if isinstance(model, KNeighborsClassifier) or isinstance(model, TanimotoKNN):
        is_knn = True
    if isinstance(model, GridSearchCV):
        if isinstance(model.estimator, KNeighborsClassifier) or isinstance(model.estimator, TanimotoKNN):
            is_knn = True

    if not is_knn:
        y_score = model.predict_proba(dataset.feature_matrix)[:, 1]
    else:
        y_score = None

    metric_list = evaluate_classification(dataset.label, y_pred, y_score)

    return metric_list


evaluate_dataset = evaluate_dataset_classification


def evaluate_classification(y_true, y_predicted, y_score=None, nantozero=False) -> pd.DataFrame:
    if len(y_true) != len(y_predicted):
        raise IndexError("y_true and y_predicted are not of equal size!")
    if y_score is not None:
        if len(y_true) != len(y_score):
            raise IndexError("y_true and y_score are not of equal size!")

    fill = 0 if nantozero else np.nan

    if sum(y_predicted) == 0:
        mcc = fill
        precision = fill
    else:
        mcc = metrics.matthews_corrcoef(y_true, y_predicted)
        precision = metrics.precision_score(y_true, y_predicted)

    result_list = [{"metric": "MCC", "value": mcc},
                   {"metric": "F1", "value": metrics.f1_score(y_true, y_predicted)},
                   {"metric": "BA", "value": metrics.balanced_accuracy_score(y_true, y_predicted)},
                   {"metric": "Precision", "value": precision},
                   {"metric": "Recall", "value": metrics.recall_score(y_true, y_predicted)},
                   {"metric": "Average Precision", "value": metrics.average_precision_score(y_true, y_predicted)},
                   {"metric": "set_size", "value": y_true.shape[0]},
                   {"metric": "pos_true", "value": len([x for x in y_true if x == 1])},
                   {"metric": "neg_true", "value": len([x for x in y_true if x == 0])},
                   {"metric": "pos_predicted", "value": len([x for x in y_predicted if x == 1])},
                   {"metric": "neg_predicted", "value": len([x for x in y_predicted if x == 0])},
                   ]

    if y_score is not None:
        result_list.append({"metric": "AUC", "value": metrics.roc_auc_score(y_true, y_score)})
    else:
        result_list.append({"metric": "AUC", "value": np.nan})

    return pd.DataFrame(result_list)


evaluate_prediction = evaluate_classification


def evaluate_regression(y_true, y_predicted) -> pd.DataFrame:
    if len(y_true) != len(y_predicted):
        raise IndexError("y_true and y_predicted are not of equal size!")

    result_list = [{"metric": "explained_variance",     "value": metrics.explained_variance_score(y_true, y_predicted)},
                   {"metric": "max_error",              "value": metrics.max_error(y_true, y_predicted)},
                   {"metric": "mean_absolute_error",    "value": metrics.mean_absolute_error(y_true, y_predicted)},
                   {"metric": "mean_squared_error",     "value": metrics.mean_squared_error(y_true, y_predicted)},
                   {"metric": "mean_squared_log_error", "value": metrics.mean_squared_log_error(y_true, y_predicted)},
                   {"metric": "median_absolute_error",  "value": metrics.median_absolute_error(y_true, y_predicted)},
                   {"metric": "r2",                     "value": metrics.r2_score(y_true, y_predicted)},
                   {"metric": "mean_poisson_deviance",  "value": metrics.mean_poisson_deviance(y_true, y_predicted)},
                   {"metric": "mean_gamma_deviance",    "value": metrics.mean_gamma_deviance(y_true, y_predicted)},
                   {"metric": "mean_absolute_percentage_error",
                    "value": metrics.mean_absolute_percentage_error(y_true, y_predicted)},
                   {"metric": "d2_absolute_error_score", "value": metrics.d2_absolute_error_score(y_true, y_predicted)},
                   {"metric": "d2_pinball_score",        "value": metrics.d2_pinball_score(y_true, y_predicted)},
                   {"metric": "d2_tweedie_score",        "value": metrics.d2_tweedie_score(y_true, y_predicted)},
                   ]

    return pd.DataFrame(result_list)



def visualize_metrics(dataframe, save_path=None, metric_list=None, figsize=(8, 6), show=True, hue="algorithm",
                      swarm=False, hue_order=None, dpi=300):
    if not metric_list:
        metric_list = ['MCC', 'F1', 'BA', 'AUC']

    zero2one_scores = {'F1', 'BA', 'AUC', 'Precision', 'Recall', "Average Precision"}
    # not using the more convinient set intersection to keep order of metric list
    zero2one_scores = [metric for metric in metric_list if metric in zero2one_scores]
    minus_one2one_scores = {'MCC', }
    minus_one2one_scores = [metric for metric in metric_list if metric in minus_one2one_scores]

    unknown_metrics = set(metric_list) - set(zero2one_scores) - set(minus_one2one_scores)

    if unknown_metrics:
        raise ValueError("Unkown metric: {}".format(", ".join(unknown_metrics)))

    n_grid_cols = round(12 * len(metric_list))
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(11, n_grid_cols, wspace=0, hspace=0)
    len_left = 10 * len(zero2one_scores)
    len_right = 10 * len(minus_one2one_scores)
    ax1 = fig.add_subplot(gs[0:9, :len_left])
    ax2 = fig.add_subplot(gs[0:9, -len_right:])
    ax3 = fig.add_subplot(gs[-1, :])

    if not hue_order:
        hue_order = sorted(dataframe[hue].unique())

    if swarm:
        vis = sns.stripplot
        kwargs = {"dodge": True}
    else:
        vis = sns.boxplot
        kwargs = {}

    left_plot = vis(data=dataframe.query("metric.isin(@zero2one_scores)"),
                    x="metric",
                    y="value",
                    hue=hue,
                    order=zero2one_scores,
                    hue_order=hue_order,
                    ax=ax1,
                    **kwargs)
    right_plot = vis(data=dataframe.query("metric.isin(@minus_one2one_scores)"),
                     x="metric",
                     y="value",
                     hue=hue,
                     order=minus_one2one_scores,
                     hue_order=hue_order,
                     ax=ax2,
                     **kwargs)

    ax2.get_legend().remove()
    ax1.legend(loc='lower left', ncol=3)

    ax1.set_ylim(-0.05, 1.05)
    ax2.set_ylim(-1.1, 1.1)

    axs_0_xlim = ax1.get_xlim()
    ax1.hlines(0.5, xmin=axs_0_xlim[0], xmax=axs_0_xlim[1], ls="--", color="gray")
    ax1.set_xlim(axs_0_xlim)

    axs_2_xlim = ax2.get_xlim()
    ax2.hlines(0, xmin=axs_2_xlim[0], xmax=axs_2_xlim[1], ls="--", color="gray")
    ax2.set_xlim(axs_2_xlim)

    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    handles, labels = ax1.get_legend_handles_labels()
    ax3.legend(handles, labels, ncol=len(handles), loc="center")
    ax1.get_legend().remove()
    ax3.axis('off')

    fig.subplots_adjust(bottom=0.0, top=0.95, right=0.95)
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    if not show:
        plt.close()
    return fig, (ax1, ax2, ax3)
