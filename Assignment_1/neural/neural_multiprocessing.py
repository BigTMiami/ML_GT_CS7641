import multiprocessing
from sklearn.model_selection import KFold
import numpy as np
from Assignment_1.neural.dqn_agent import DQNAgent
from Assignment_1.neural.dqn_multi_module import run_training
from Assignment_1.prep_census_data import get_census_data_and_labels
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


def process_results(results):
    print(results)
    print("============================================================")
    print(f"Results")

    result_set = []
    for result in results:
        a = np.array(result[1])
        result_set.append(a)

    all_scores = np.array(result_set)
    avg_scores = np.average(all_scores, axis=0)

    max_indexes = np.argmax(avg_scores, axis=0)

    best_cv_epoch = max_indexes[3]
    best_cv_score = avg_scores[best_cv_epoch][3]

    best_test_epoch = max_indexes[2]
    best_test_score = avg_scores[best_test_epoch][2]

    print(f"        Best CV Error Epoch:{best_cv_epoch:2} value:{best_cv_score:8.5}%")
    print(f"FINAL Best Test Error Epoch:{best_test_epoch:2} value:{best_test_score:8.5}%")
    print(f"  Best Training Error Epoch:{max_indexes[1]:2} value:{avg_scores[max_indexes[1]][1]:8.5}%")

    return best_cv_epoch, best_cv_score, best_test_epoch, best_test_score, avg_scores


def process_premutation_results(results):
    results = np.array(results)
    sorted_results = sorted(results, key=lambda x: x[6], reverse=True)
    for (
        network_learning_rate,
        layer_one_size,
        epoch_count,
        best_cv_epoch,
        best_cv_score,
        best_test_epoch,
        best_test_score,
    ) in sorted_results:
        print(f"{network_learning_rate:0.6f} {layer_one_size} {epoch_count} {best_cv_epoch:3} {best_test_score:9.6f} ")

    fig, ax = plt.subplots(1, figsize=(4, 5))
    fig.suptitle("Census Data Neural Network", fontsize=16)
    ax.set_title("Error Rates by Layer Size, Learning Rate")
    for lr in np.unique(results[:, 0]):
        filtered_results = results[(lr == results[:, 0]) & (200 == results[:, 2])]
        x = filtered_results[:, 1]
        y = 100 - filtered_results[:, 6]
        ax.plot(x, y, label=f"Learning Rate {lr}")
    ax.set_xlabel("Layer Size")
    ax.set_ylabel("Error")
    plt.legend()

    filename = title_to_filename("Census Data Neural Network Error Rates by Layer Size, Learning Rate", "census")
    if os.path.exists(filename):
        os.remove(filename)
    fig.tight_layout()
    plt.savefig(fname=filename, bbox_inches="tight")


def title_to_filename(title, classifier_type):
    safe_title = title.replace(" ", "_")
    safe_title = safe_title.replace(":", "_")
    safe_title = safe_title.replace(",", "_")
    return f"Assignment_1/document/figures/{safe_title}_{classifier_type}.png"


def plot_epoch_error(avg_scores, start_node=5, title="Census", subtitle="Error"):
    avg_error = 100.0 - avg_scores
    fig, ax = plt.subplots(1, figsize=(4, 5))
    fig.suptitle(title, fontsize=16)
    x = list(range(start_node, len(avg_scores)))
    ax.set_title(subtitle)
    ax.plot(x, avg_error[start_node:, 1], label="Training Data")
    ax.plot(x, avg_error[start_node:, 3], label="Cross Val Data")
    ax.plot(x, avg_error[start_node:, 2], label="Test Data")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    plt.legend()

    filename = title_to_filename(title + " " + subtitle, "census")
    if os.path.exists(filename):
        os.remove(filename)
    fig.tight_layout()
    plt.savefig(fname=filename, bbox_inches="tight")


def train_with_cv(**kwargs):
    training_data = kwargs["training_data"]
    all_indexes = list(range(len(training_data)))
    kf = KFold(n_splits=4)
    split_count = 0

    all_training_sets = []
    print("============================================================")
    print(f"Loading Splits ")

    for train_indexes, test_indexes in kf.split(all_indexes):

        dqn = DQNAgent(**kwargs)
        training_set = (split_count, dqn, train_indexes, test_indexes, True)

        all_training_sets.append(training_set)
        # epoch_values = run_training(training_set)
        split_count += 1

    print("============================================================")
    print(f"Running Splits ")
    with multiprocessing.Pool(processes=4) as pool:
        # results = pool.starmap(train_test_model, training_iterations)
        results = pool.starmap(run_training, all_training_sets)

    return results


def train_permutations(**kwargs):
    network_learning_rate_set = kwargs["network_learning_rate_set"]
    layer_one_size_set = kwargs["layer_one_size_set"]
    epoch_count_set = kwargs["epoch_count_set"]

    all_scores = []

    for network_learning_rate in network_learning_rate_set:
        for layer_one_size in layer_one_size_set:
            for epoch_count in epoch_count_set:
                print(f"Starting {network_learning_rate},{layer_one_size},{epoch_count}")
                results = train_with_cv(
                    network_learning_rate=network_learning_rate,
                    layer_one_size=layer_one_size,
                    epoch_count=epoch_count,
                    **kwargs,
                )
                best_cv_epoch, best_cv_score, best_test_epoch, best_test_score, epoch_scores = process_results(results)

                all_scores.append(
                    [
                        network_learning_rate,
                        layer_one_size,
                        epoch_count,
                        best_cv_epoch,
                        best_cv_score,
                        best_test_epoch,
                        best_test_score,
                    ]
                )

                title = "Census Data Training Error"
                subtitle = f"Learning Rate:{network_learning_rate}, Layer Size:{layer_one_size}, Epochs:{epoch_count}"
                plot_epoch_error(epoch_scores, title=title, subtitle=subtitle)

    return all_scores


if __name__ == "main":
    # Census Data
    (
        df_data,
        df_label,
        df_data_numeric,
        df_label_numeric,
        df_test_data,
        df_test_label,
        df_test_data_numeric,
        df_test_label_numeric,
        data_classes,
    ) = get_census_data_and_labels()

    scaler = StandardScaler()
    scaler.fit(df_data_numeric.to_numpy())
    scaled_training_data = scaler.transform(df_data_numeric.to_numpy())
    scaled_test_data = scaler.transform(df_test_data_numeric.to_numpy())

    results = train_with_cv(
        training_data=scaled_training_data,
        training_labels=df_label_numeric.to_numpy(),
        test_data=scaled_test_data,
        test_labels=df_test_label_numeric.to_numpy(),
        network_learning_rate=0.00005,
        layer_one_size=10,
        epoch_count=100,
    )

    best_cv_epoch, best_test_score, epoch_scores = process_results(results)

    plot_epoch_error(epoch_scores)

    permutation_results = train_permutations(
        training_data=scaled_training_data,
        training_labels=df_label_numeric.to_numpy(),
        test_data=scaled_test_data,
        test_labels=df_test_label_numeric.to_numpy(),
        network_learning_rate_set=[0.00001, 0.00002, 0.00004, 0.00008],
        layer_one_size_set=[10, 20, 40, 80],
        epoch_count_set=[100, 200],
    )

    process_premutation_results(results)
