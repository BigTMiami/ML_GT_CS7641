from sklearn.model_selection import KFold
from Assignment_1.neural.mnist_network_model import MNISTNet, MNISTData
from Assignment_1.neural.mnist_network_model_cnn import MNISTNetCNN
import multiprocessing
from torch.utils.data import DataLoader
from Assignment_1.neural.mnist_multi_module import run_training
from Assignment_1.mnist_data_prep import get_mnist_data_labels_neural
import numpy as np
import matplotlib.pyplot as plt
import os


def title_to_filename(title, classifier_type):
    safe_title = title.replace(" ", "_")
    safe_title = safe_title.replace(":", "_")
    safe_title = safe_title.replace(",", "_")
    return f"Assignment_1/document/figures/working/{safe_title}_{classifier_type}.png"


def plot_epoch_error(avg_scores, start_node=0, title="MNIST", subtitle="Error"):
    avg_error = 100.0 - avg_scores
    fig, ax = plt.subplots(1, figsize=(4, 5))
    fig.suptitle(title, fontsize=16)
    x = list(range(start_node, len(avg_scores)))
    ax.set_title(subtitle)
    # ax.plot(x, avg_error[start_node:, 1], label="Training Data")
    ax.plot(x, avg_error[start_node:, 2], label="Cross Val Data")
    ax.plot(x, avg_error[start_node:, 3], label="Test Data")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    plt.legend()

    filename = title_to_filename(title + " " + subtitle, "MNIST")
    if os.path.exists(filename):
        os.remove(filename)
    fig.tight_layout()
    plt.savefig(fname=filename, bbox_inches="tight")


def process_results(results):
    # epoch, running_loss, cv_acc, test_acc
    print("============================================================")
    print(f"Results")

    result_set = []
    for result in results:
        a = np.array(result[1])
        result_set.append(a)

    all_scores = np.array(result_set)
    avg_scores = np.average(all_scores, axis=0)

    max_indexes = np.argmax(avg_scores, axis=0)

    best_cv_epoch = max_indexes[2]
    best_cv_score = avg_scores[best_cv_epoch][2]

    best_test_epoch = max_indexes[3]
    best_test_score = avg_scores[best_test_epoch][3]

    print(f"        Best CV Error Epoch:{best_cv_epoch:2} value:{best_cv_score:8.5}%")
    print(f"FINAL Best Test Error Epoch:{best_test_epoch:2} value:{best_test_score:8.5}%")
    # print(f"  Best Training Error Epoch:{max_indexes[1]:2} value:{avg_scores[max_indexes[1]][1]:8.5}%")

    return best_cv_epoch, best_cv_score, best_test_epoch, best_test_score, avg_scores


def train_with_cv(**kwargs):
    train_data = kwargs["train_data"]
    train_one_hot_labels = kwargs["train_one_hot_labels"]
    train_labels = kwargs["train_labels"]
    test_data = kwargs["test_data"]
    test_labels = kwargs["test_labels"]
    epoch_count = kwargs["epoch_count"]
    use_cnn = kwargs["use_cnn"]
    all_indexes = list(range(len(train_data)))
    kf = KFold(n_splits=4)
    split_count = 0

    all_training_sets = []
    print("============================================================")
    print(f"Loading Splits ")

    for train_indexes, test_indexes in kf.split(all_indexes):
        mnist = MNISTData(train_data[train_indexes], train_one_hot_labels[train_indexes])
        cv_data = train_data[test_indexes]
        cv_labels = train_labels[test_indexes]
        mnist_loader = DataLoader(mnist, batch_size=100, shuffle=True)

        if use_cnn:
            model = MNISTNetCNN(
                training_data_loader=mnist_loader,
                test_data=test_data,
                test_labels=test_labels,
                cv_data=cv_data,
                cv_labels=cv_labels,
                epoch_count=epoch_count,
            )
        else:
            model = MNISTNet(
                training_data_loader=mnist_loader,
                test_data=test_data,
                test_labels=test_labels,
                cv_data=cv_data,
                cv_labels=cv_labels,
                epoch_count=epoch_count,
            )
        training_set = (split_count, model)

        all_training_sets.append(training_set)
        split_count += 1

    print("============================================================")
    print(f"Running Splits ")
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(run_training, all_training_sets)

    return results


if __name__ == "main":
    (
        train_data,
        train_one_hot_labels,
        train_labels,
        test_images,
        test_one_hot_labels,
        test_labels,
    ) = get_mnist_data_labels_neural(flatten_images=False)

    model = MNISTNetCNN(
        train_data=train_data,
        train_one_hot_labels=train_one_hot_labels,
        train_labels=train_labels,
        test_data=test_images,
        test_labels=test_labels,
        cv_data=test_images,
        cv_labels=test_labels,
        epoch_count=10,
        batch_size=100,
    )
    results = model.train()

    best_cv_epoch, best_cv_score, best_test_epoch, best_test_score, avg_scores = process_results([[0, results]])

    plot_epoch_error(avg_scores)

    results = train_with_cv(
        train_data=train_data,
        train_one_hot_labels=train_one_hot_labels,
        train_labels=train_labels,
        test_data=test_images,
        test_labels=test_labels,
        epoch_count=10,
        use_cnn=True,
    )

    best_cv_epoch, best_cv_score, best_test_epoch, best_test_score, avg_scores = process_results(results)

    plot_epoch_error(avg_scores)

    if False:
        (
            train_images_flattened,
            train_one_hot_labels,
            train_labels,
            test_images_flattened,
            test_one_hot_labels,
            test_labels,
        ) = get_mnist_data_labels_neural()

        results = train_with_cv(
            train_data=train_images_flattened,
            train_one_hot_labels=train_one_hot_labels,
            train_labels=train_labels,
            test_data=test_images_flattened,
            test_labels=test_labels,
            epoch_count=10,
            use_cnn=False,
        )

        best_cv_epoch, best_cv_score, best_test_epoch, best_test_score, avg_scores = process_results(results)

        plot_epoch_error(avg_scores)
