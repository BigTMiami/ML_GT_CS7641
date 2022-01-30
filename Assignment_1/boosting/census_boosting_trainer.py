from sklearn.ensemble import GradientBoostingClassifier
from Assignment_1.prep_census_data import get_census_data_and_labels
from Assignment_1.mnist_data_prep import get_mnist_data_labels_neural
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import os


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

    n_estimators = kwargs["n_estimators"]
    max_features = kwargs["max_features"]
    max_depth = kwargs["max_depth"]
    loss_function = kwargs["loss_function"]

    use_cv = kwargs["use_cv"]
    cv_count = kwargs["cv_count"]

    gb_clf = GradientBoostingClassifier(
        loss=loss_function,
        learning_rate=1,
        random_state=0,
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
    )

    if use_cv:
        cv_score = cross_val_score(gb_clf, train_data, train_labels, cv=cv_count).mean()
    else:
        cv_score = 0.0

    gb_clf.fit(train_data, train_labels)
    train_score = gb_clf.score(train_data, train_labels)
    test_score = gb_clf.score(test_data, test_labels)

    return train_score, cv_score, test_score


def train_permutations(**kwargs):
    n_estimators_set = kwargs["n_estimators_set"]
    max_features_set = kwargs["max_features_set"]
    max_depth_set = kwargs["max_depth_set"]

    all_scores = []

    for n_estimators in n_estimators_set:
        for max_features in max_features_set:
            for max_depth in max_depth_set:
                print(f"Starting {n_estimators},{max_features},{max_depth}")
                train_score, cv_score, test_score = train_with_cv(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    max_depth=max_depth,
                    **kwargs,
                )
                print(
                    f"{n_estimators}, {max_features}, {max_depth}:  test:{test_score*100:6.3f} cv:{cv_score*100:6.3f} train:{train_score*100:6.3f}"
                )
                all_scores.append([n_estimators, max_features, max_depth, train_score, cv_score, test_score])

    return all_scores


def show_permutation_results(results, x_col, line_col, title, subtitle):
    columns = ["n_estimators", "max_features", "max_depth", "train_score", "cv_score", "test_score"]

    x_index = columns.index(x_col)
    line_index = columns.index(line_col)

    r = np.array(results)
    rs = r[:, 5].argsort(axis=0)[::-1]
    r_show = r[rs, :]
    for n_estimators, max_features, max_depth, train_score, cv_score, test_score in r_show:
        print(
            f"{n_estimators}, {max_features}, {max_depth}:  test:{test_score*100:6.3f} cv:{cv_score*100:6.3f} train:{train_score*100:6.3f}"
        )

    fig, ax = plt.subplots(1, figsize=(4, 5))
    fig.suptitle(title, fontsize=16)
    ax.set_title(subtitle)
    for line_value in np.unique(r[:, line_index]):
        filtered_results = r[(line_value == r[:, line_index])]
        fr_sort = filtered_results[:, x_index].argsort()
        x = filtered_results[fr_sort, x_index]
        y = 1 - filtered_results[fr_sort, 5]
        ax.plot(x, y, label=f"{line_col} {line_value}")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Error")
    plt.legend()

    filename = title_to_filename(title, "boosting")
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

    train_score, cv_score, test_score = train_with_cv(
        train_data=np_train_data_numeric,
        train_labels=np_train_label_numeric.ravel(),
        test_data=np_test_data_numeric,
        test_labels=np_test_label_numeric.ravel(),
        n_estimators=5,
        max_features=2,
        max_depth=3,
        cv_count=4,
    )
    print(f"test:{test_score*100:6.3f} cv:{cv_score*100:6.3f} train:{train_score*100:6.3f}")

    est_vs_depth_results = train_permutations(
        train_data=np_train_data_numeric,
        train_labels=np_train_label_numeric.ravel(),
        test_data=np_test_data_numeric,
        test_labels=np_test_label_numeric.ravel(),
        n_estimators_set=[5, 10, 20, 40, 80, 160],
        max_features_set=[3],
        max_depth_set=[2, 3, 5, 10, 20],
        cv_count=4,
    )

    est_vs_depth_results

    show_permutation_results(
        est_vs_depth_results,
        x_col="n_estimators",
        line_col="max_depth",
        title="Census Data Boosted Ensemble",
        subtitle="Error Rates by Estimator Count, Max Depth",
    )

    est_vs_features = train_permutations(
        train_data=np_train_data_numeric,
        train_labels=np_train_label_numeric.ravel(),
        test_data=np_test_data_numeric,
        test_labels=np_test_label_numeric.ravel(),
        n_estimators_set=[40, 80, 160, 320],
        max_features_set=[2, 3, 6, 12, 20],
        max_depth_set=[3],
        cv_count=4,
    )

    est_vs_features

    show_permutation_results(
        est_vs_features,
        x_col="n_estimators",
        line_col="max_features",
        title="Census Data Boosted Ensemble",
        subtitle="Error Rates by Estimator Count, Max Features",
    )

    long_est_vs_tight_features = train_permutations(
        train_data=np_train_data_numeric,
        train_labels=np_train_label_numeric.ravel(),
        test_data=np_test_data_numeric,
        test_labels=np_test_label_numeric.ravel(),
        n_estimators_set=[100, 200, 400, 800, 1600],
        max_features_set=[5, 6, 7],
        max_depth_set=[3],
        cv_count=4,
        use_cv=False,
    )

    long_est_vs_tight_features

    show_permutation_results(
        long_est_vs_tight_features,
        x_col="n_estimators",
        line_col="max_features",
        title="Census Data Boosted Ensemble",
        subtitle="Error Rates by Estimator Count, Max Features",
    )

    final_est_vs_tight_features = train_permutations(
        train_data=np_train_data_numeric,
        train_labels=np_train_label_numeric.ravel(),
        test_data=np_test_data_numeric,
        test_labels=np_test_label_numeric.ravel(),
        n_estimators_set=[100, 150, 200, 250, 300, 350, 400],
        max_features_set=[4, 5, 6],
        max_depth_set=[3],
        cv_count=4,
        use_cv=False,
    )

    final_est_vs_tight_features

    show_permutation_results(
        final_est_vs_tight_features,
        x_col="n_estimators",
        line_col="max_features",
        title="Census Data Boosted Ensemble",
        subtitle="Error Rates by Estimator Count, Max Features",
    )

    # MNIST

    (
        mn_train_data,
        mn_train_one_hot_labels,
        mn_train_labels,
        mn_test_images,
        mn_test_one_hot_labels,
        mn_test_labels,
    ) = get_mnist_data_labels_neural(flatten_images=True)

    mn_est_vs_depth_results = train_permutations(
        train_data=mn_train_data,
        train_labels=mn_train_labels,
        test_data=mn_test_images,
        test_labels=mn_test_labels,
        n_estimators_set=[5, 10, 20, 40, 80, 160, 320],
        max_features_set=[28],
        max_depth_set=[2, 4, 7, 11, 15],
        cv_count=4,
        use_cv=False,
        loss_function="deviance",
    )

    mn_est_vs_depth_results

    show_permutation_results(
        mn_est_vs_depth_results,
        x_col="n_estimators",
        line_col="max_depth",
        title="MNIST Boosted Ensemble",
        subtitle="Error Rates by Estimator Count, Max Depth",
    )

    mn_feature_vs_depth_results = train_permutations(
        train_data=mn_train_data,
        train_labels=mn_train_labels,
        test_data=mn_test_images,
        test_labels=mn_test_labels,
        n_estimators_set=[25],
        max_features_set=[18, 23, 28, 33, 38],
        max_depth_set=[2, 4, 7, 11, 15],
        cv_count=4,
        use_cv=False,
        loss_function="deviance",
    )

    mn_feature_vs_depth_results

    show_permutation_results(
        mn_feature_vs_depth_results,
        x_col="max_features",
        line_col="max_depth",
        title="MNIST Boosted Ensemble",
        subtitle="Error Rates by Max Features, Max Depth",
    )

    mn_feature_vs_depth_results_2 = train_permutations(
        train_data=mn_train_data,
        train_labels=mn_train_labels,
        test_data=mn_test_images,
        test_labels=mn_test_labels,
        n_estimators_set=[25],
        max_features_set=[28, 30, 33, 38],
        max_depth_set=[13, 14, 15, 16, 17],
        cv_count=4,
        use_cv=False,
        loss_function="deviance",
    )

    mn_feature_vs_depth_results_2

    show_permutation_results(
        mn_feature_vs_depth_results_2,
        x_col="max_features",
        line_col="max_depth",
        title="MNIST Boosted Ensemble",
        subtitle="Error Rates by Max Features, Max Depth",
    )
