from Assignment_1.prep_census_data import get_census_data_and_labels
from Assignment_1.mnist_data_prep import get_mnist_data_labels_neural
from Assignment_1.neural.mnist_network_model_cnn import MNISTNetCNN
from sklearn import tree
import torch as t
from time import time

random_state = 1


def review_model(model, train_data, train_labels, test_data, test_labels, convert_one_hot_labels=False):
    fit_start = time()
    model.fit(train_data, train_labels)
    fit_time = time() - fit_start

    predict_start = time()
    predictions = model.predict(test_data)
    predict_time = time() - predict_start

    if convert_one_hot_labels:
        _, predictions = t.max(predictions, 1)
    correct = (predictions == test_labels).sum().float()
    acc = 100 * correct / len(test_labels)

    return acc, fit_time, predict_time


if __name__ == "main":
    (
        census_train_data,
        census_train_label,
        census_train_data_numeric,
        census_train_label_numeric,
        census_test_data,
        census_test_label,
        census_test_data_numeric,
        census_test_label_numeric,
        data_classes,
    ) = get_census_data_and_labels(scale_numeric=True)

    (
        mn_train_data,
        mn_train_one_hot_labels,
        mn_train_labels,
        mn_test_images,
        mn_test_one_hot_labels,
        mn_test_labels,
    ) = get_mnist_data_labels_neural(flatten_images=True)

    results = []

    # Decision Tree
    census_best_alpha = 0.0000153
    census_tree = tree.DecisionTreeClassifier(ccp_alpha=census_best_alpha, random_state=random_state)
    acc, fit_time, predict_time = review_model(
        census_tree,
        census_train_data_numeric,
        census_train_label_numeric,
        census_test_data_numeric,
        census_test_label_numeric,
    )
    results.append("Decision Tree", "Census", "ccp_alpha = 0.0000153", acc, fit_time, predict_time)

    mnist_best_alpha = 0.0000762
    mnist_tree = tree.DecisionTreeClassifier(ccp_alpha=mnist_best_alpha, random_state=random_state)
    acc, fit_time, predict_time = review_model(
        mn_train_data,
        mn_train_labels,
        mn_test_images,
        mn_test_labels,
    )
    results.append("Decision Tree", "MNIST", "ccp_alpha = 0.0000762", acc, fit_time, predict_time)

    # Neural Network
