from sklearn.model_selection import KFold
from Assignment_1.neural.mnist_network_model import MNISTNet, MNISTData
import multiprocessing
from torch.utils.data import DataLoader
from Assignment_1.neural.mnist_multi_module import run_training
from Assignment_1.mnist_data_prep import get_mnist_data_labels_neural
import numpy as np


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
        epoch_count=100,
    )

    best_cv_epoch, best_cv_score, best_test_epoch, best_test_score, avg_scores = process_results(results)
