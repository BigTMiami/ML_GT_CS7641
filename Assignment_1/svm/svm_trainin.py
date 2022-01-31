from sklearn.svm import SVC
from Assignment_1.prep_census_data import get_census_data_and_labels
from Assignment_1.mnist_data_prep import get_mnist_data_labels_neural
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
from Assignment_1.svm.run_model import train_with_cv_mulit


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

    kernel = kwargs["kernel"]

    clf = SVC(random_state=0, kernel=kernel)

    if use_cv:
        cv_score = cross_val_score(clf, train_data, train_labels, cv=cv_count).mean()
    else:
        cv_score = 0.0

    clf.fit(train_data, train_labels)
    train_score = clf.score(train_data, train_labels)
    test_score = clf.score(test_data, test_labels)

    return train_score, cv_score, test_score


def train_permutations(**kwargs):
    kernel_set = kwargs["kernel_set"]
    all_scores = []
    for kernel in kernel_set:

        print(f"Starting {kernel}")
        train_score, cv_score, test_score = train_with_cv(
            kernel=kernel,
            **kwargs,
        )
        # Make a beep
        print("\a")
        print(f"{kernel}:  test:{test_score*100:6.3f} cv:{cv_score*100:6.3f} train:{train_score*100:6.3f}")
        all_scores.append([kernel, train_score, cv_score, test_score])

    return all_scores


def train_permutations_multi(**kwargs):
    kernel_set = kwargs["kernel_set"]
    training_sets = []
    train_data = kwargs["train_data"]
    train_labels = kwargs["train_labels"]

    test_data = kwargs["test_data"]
    test_labels = kwargs["test_labels"]

    use_cv = kwargs["use_cv"]
    cv_count = kwargs["cv_count"]

    training_sets = []
    for kernel in kernel_set:
        model = SVC(random_state=0, kernel=kernel)
        training_sets.append([kernel, train_data, train_labels, test_data, test_labels, use_cv, cv_count, model])

    print("============================================================")
    print(f"Running Splits ")
    with multiprocessing.Pool(processes=4) as pool:
        # results = pool.starmap(train_test_model, training_iterations)
        results = pool.starmap(train_with_cv_mulit, training_sets)

    return results


def show_permutation_results_bar(results, x_col, title, subtitle):
    columns = ["kernel", "train_score", "cv_score", "test_score"]

    r = np.array(results)
    rs = r[:, 3].argsort(axis=0)[::-1]
    r_show = r[rs, :]
    for kernel, train_score, cv_score, test_score in r_show:
        print(
            f"{kernel:>10}:  test:{float(test_score)*100:6.3f} cv:{float(cv_score)*100:6.3f} train:{float(train_score)*100:6.3f}"
        )

    fig, ax = plt.subplots(1, figsize=(4, 5))
    fig.suptitle(title, fontsize=16)
    ax.set_title(subtitle)
    x = r[:, 0]
    y = 100 * (1 - r[:, 3].astype(float))
    plt.bar(x, y)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Error %")
    ax.set_ylim(0, 9)
    plt.legend()

    filename = title_to_filename(title, "svm")
    if os.path.exists(filename):
        os.remove(filename)
    fig.tight_layout()
    plt.savefig(fname=filename, bbox_inches="tight")


if __name__ == "main":
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

    results = train_with_cv(
        train_data=np_train_data_numeric,
        train_labels=np_train_label_numeric.ravel(),
        test_data=np_test_data_numeric,
        test_labels=np_test_label_numeric.ravel(),
        cv_count=4,
        use_cv=False,
    )

    print(results)

    census_svc_results = train_permutations_multi(
        train_data=np_train_data_numeric,
        train_labels=np_train_label_numeric.ravel(),
        test_data=np_test_data_numeric,
        test_labels=np_test_label_numeric.ravel(),
        cv_count=4,
        use_cv=False,
        kernel_set=["linear", "poly", "rbf", "sigmoid"],
    )

    show_permutation_results_bar(
        census_svc_results,
        x_col="kernel",
        title="Census Data SVM",
        subtitle="Error Rates by Kernel Type",
    )

    (
        mn_train_data,
        mn_train_one_hot_labels,
        mn_train_labels,
        mn_test_images,
        mn_test_one_hot_labels,
        mn_test_labels,
    ) = get_mnist_data_labels_neural(flatten_images=True)

    mnist_svc_results = train_permutations_multi(
        train_data=mn_train_data,
        train_labels=mn_train_labels.ravel(),
        test_data=mn_test_images,
        test_labels=mn_test_labels.ravel(),
        cv_count=4,
        use_cv=False,
        kernel_set=["linear", "poly", "rbf", "sigmoid"],
    )

    show_permutation_results_bar(
        mnist_svc_results,
        x_col="kernel",
        title="MNIST Image Data SVM",
        subtitle="Error Rates by Kernel Type",
    )
