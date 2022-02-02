from sklearn.neighbors import KNeighborsClassifier
from Assignment_1.prep_census_data import get_census_data_and_labels
from Assignment_1.prep_census_data import get_census_data_and_labels_one_hot
from Assignment_1.mnist_data_prep import get_mnist_data_labels_neural
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
from Assignment_1.knn.knn_run_model import train_with_cv_multi


def title_to_filename(title, classifier_type):
    safe_title = title.replace(" ", "_")
    safe_title = safe_title.replace(":", "_")
    safe_title = safe_title.replace(",", "_")
    return f"Assignment_1/document/figures/working/{safe_title}_{classifier_type}.png"


def train_with_cv(**kwargs):
    train_data = kwargs["train_data"]
    train_labels = kwargs["train_labels"]

    test_data = kwargs["test_data"]
    test_labels = kwargs["test_labels"]

    use_cv = kwargs["use_cv"]
    cv_count = kwargs["cv_count"]

    n_neighbors = kwargs["n_neighbors"]
    weights = kwargs["weights"]
    metric = kwargs["metric"]

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

    if use_cv:
        cv_score = cross_val_score(clf, train_data, train_labels, cv=cv_count).mean()
    else:
        cv_score = 0.0

    clf.fit(train_data, train_labels)
    train_score = clf.score(train_data, train_labels)
    test_score = clf.score(test_data, test_labels)

    return train_score, cv_score, test_score


def train_permutations(**kwargs):
    n_neighbors_set = kwargs["n_neighbors_set"]
    weights_set = kwargs["weights_set"]
    metric_set = kwargs["metric_set"]

    all_scores = []
    for n_neighbors in n_neighbors_set:
        for weights in weights_set:
            for metric in metric_set:
                print(f"Starting {n_neighbors},{weights}, {metric}  ")
                train_score, cv_score, test_score = train_with_cv(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric,
                    **kwargs,
                )
                # Make a beep
                print("\a")
                print(
                    f"{n_neighbors},{weights}, {metric}:  test:{test_score*100:6.3f} cv:{cv_score*100:6.3f} train:{train_score*100:6.3f}"
                )
                all_scores.append([n_neighbors, weights, metric, train_score, cv_score, test_score])

    return all_scores


def train_permutations_multi(**kwargs):
    n_neighbors_set = kwargs["n_neighbors_set"]
    weights_set = kwargs["weights_set"]
    metric_set = kwargs["metric_set"]

    train_data = kwargs["train_data"]
    train_labels = kwargs["train_labels"]

    test_data = kwargs["test_data"]
    test_labels = kwargs["test_labels"]

    use_cv = kwargs["use_cv"]
    cv_count = kwargs["cv_count"]

    training_sets = []
    for n_neighbors in n_neighbors_set:
        for weights in weights_set:
            for metric in metric_set:
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
                training_sets.append(
                    [
                        n_neighbors,
                        weights,
                        metric,
                        train_data,
                        train_labels,
                        test_data,
                        test_labels,
                        use_cv,
                        cv_count,
                        model,
                    ]
                )

    print("============================================================")
    print(f"Running Splits ")
    with multiprocessing.Pool(processes=4) as pool:
        # results = pool.starmap(train_test_model, training_iterations)
        results = pool.starmap(train_with_cv_multi, training_sets)

    return results


def show_permutation_results_bar(results, x_col, line_col, title, subtitle):
    columns = ["n_neighbors", "weights", "metric", "train_score", "cv_score", "test_score"]
    x_index = columns.index(x_col)
    line_index = columns.index(line_col)

    r = np.array(results)
    rs = r[:, 5].argsort(axis=0)[::-1]
    r_show = r[rs, :]
    for n_neighbors, weights, metric, train_score, cv_score, test_score in r_show:
        print(
            f"{n_neighbors:3} {weights:>10} {metric:>10}:  test:{float(test_score)*100:6.3f} cv:{float(cv_score)*100:6.3f} train:{float(train_score)*100:6.3f}"
        )

    fig, ax = plt.subplots(1, figsize=(4, 5))
    fig.suptitle(title, fontsize=16)
    ax.set_title(subtitle)
    for line_value in np.unique(r[:, line_index]):
        filtered_results = r[(line_value == r[:, line_index])]

        x = filtered_results[:, x_index].astype(int)
        y = 100 * (1 - filtered_results[:, 5].astype(float))
        fr_sort = filtered_results[:, x_index].argsort()
        ax.plot(x, y, label=f"{line_col} {line_value}")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Error %")
    # ax.set_ylim(0, 9)
    plt.legend()

    filename = title_to_filename(title, "knn")
    if os.path.exists(filename):
        os.remove(filename)
    fig.tight_layout()
    plt.savefig(fname=filename, bbox_inches="tight")


if __name__ == "main":

    (
        train_data_one_hot,
        train_label_one_hot,
        test_data_one_hot,
        test_label_one_hot,
    ) = get_census_data_and_labels_one_hot()

    census_knn_results_one_hot = train_permutations_multi(
        train_data=train_data_one_hot,
        train_labels=train_label_one_hot.ravel(),
        test_data=test_data_one_hot,
        test_labels=test_label_one_hot.ravel(),
        cv_count=4,
        use_cv=False,
        n_neighbors_set=[15, 17, 21, 25, 31],
        weights_set=["uniform", "distance"],
        metric_set=["minkowski"],
    )

    (
        df_train_data,
        df_train_label,
        np_train_data_numeric,
        np_train_label_numeric,
        df_test_data,
        df_test_label,
        np_test_data_numeric,
        np_test_label_numeric,
        data_classes,
    ) = get_census_data_and_labels(scale_numeric=True)

    census_knn_results_4 = train_permutations_multi(
        train_data=np_train_data_numeric,
        train_labels=np_train_label_numeric.ravel(),
        test_data=np_test_data_numeric,
        test_labels=np_test_label_numeric.ravel(),
        cv_count=4,
        use_cv=False,
        n_neighbors_set=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        weights_set=["uniform"],
        metric_set=["minkowski", "hamming", "manhattan"],
    )

    census_knn_results_3 = train_permutations_multi(
        train_data=np_train_data_numeric,
        train_labels=np_train_label_numeric.ravel(),
        test_data=np_test_data_numeric,
        test_labels=np_test_label_numeric.ravel(),
        cv_count=4,
        use_cv=False,
        n_neighbors_set=[15, 17, 21, 25, 31],
        weights_set=["uniform", "distance"],
        metric_set=["minkowski"],
    )

    (
        mn_train_data,
        mn_train_one_hot_labels,
        mn_train_labels,
        mn_test_images,
        mn_test_one_hot_labels,
        mn_test_labels,
    ) = get_mnist_data_labels_neural(flatten_images=True)

    mnist_knn_results_1 = train_permutations_multi(
        train_data=mn_train_data,
        train_labels=mn_train_labels.ravel(),
        test_data=mn_test_images,
        test_labels=mn_test_labels.ravel(),
        cv_count=4,
        use_cv=False,
        n_neighbors_set=[5, 10, 20, 40, 80, 160],
        weights_set=["uniform", "distance"],
        metric_set=["minkowski"],
    )

    show_permutation_results_bar(
        mnist_knn_results_1,
        x_col="n_neighbors",
        line_col="weights",
        title="MNIST Images KNN",
        subtitle="Error Rates by K neighbors, Distance Weighting",
    )

    mnist_knn_results_2 = train_permutations_multi(
        train_data=mn_train_data,
        train_labels=mn_train_labels.ravel(),
        test_data=mn_test_images,
        test_labels=mn_test_labels.ravel(),
        cv_count=4,
        use_cv=False,
        n_neighbors_set=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        weights_set=["distance", "uniform"],
        metric_set=["minkowski"],
    )

    show_permutation_results_bar(
        mnist_knn_results_2,
        x_col="n_neighbors",
        line_col="weights",
        title="MNIST Images KNN",
        subtitle="Error Rates by K neighbors, Distance Weighting",
    )
