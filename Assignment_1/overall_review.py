from Assignment_1.prep_census_data import get_census_data_and_labels
from Assignment_1.mnist_data_prep import get_mnist_data_labels_neural
from Assignment_1.neural.mnist_network_model_cnn import MNISTNetCNN
from Assignment_1.neural.dqn_agent import DQNAgent
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import torch as t
from time import time
import numpy as np

random_state = 1


def review_model(model, train_data, train_labels, test_data, test_labels, neural_net=False, census_net=False):
    fit_start = time()
    model.fit(train_data, train_labels)
    fit_time = time() - fit_start

    predict_start = time()
    predictions = model.predict(test_data)
    predict_time = time() - predict_start

    if neural_net:
        _, predictions = t.max(predictions, 1)
    elif census_net:
        predictions = predictions.numpy()
    else:
        predictions = np.reshape(predictions, (len(predictions), 1))

    correct = (predictions == test_labels).sum()
    total = len(test_labels)
    acc = 100.0 * correct / total

    print(f"acc:{acc:6.3f} fit:{fit_time:6.3f} predict{predict_time:6.3f}")
    return acc, fit_time, predict_time


if __name__ == "main":
    # Load Data
    ################################################################
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
    ################################################################
    # Census
    census_best_alpha = 0.0000153
    census_tree = tree.DecisionTreeClassifier(ccp_alpha=census_best_alpha, random_state=random_state)
    acc, fit_time, predict_time = review_model(
        census_tree,
        census_train_data_numeric,
        census_train_label_numeric,
        census_test_data_numeric,
        census_test_label_numeric,
    )
    results.append(["Decision Tree", "Census", "ccp_alpha = 0.0000153", acc, fit_time, predict_time])

    # MNIST
    mnist_best_alpha = 0.0000762
    mnist_tree = tree.DecisionTreeClassifier(ccp_alpha=mnist_best_alpha, random_state=random_state)
    acc, fit_time, predict_time = review_model(
        mnist_tree,
        mn_train_data,
        np.reshape(mn_train_labels.numpy(), (len(mn_train_labels), 1)),
        mn_test_images,
        np.reshape(mn_test_labels.numpy(), (len(mn_test_labels), 1)),
    )
    results.append(["Decision Tree", "MNIST", "ccp_alpha = 0.0000762", acc, fit_time, predict_time])

    # Neural Network
    ################################################################
    # Census
    network_learning_rate = 0.000001
    layer_one_size = 16
    epoch_count = 275

    census_net = DQNAgent(
        network_learning_rate=network_learning_rate,
        layer_one_size=layer_one_size,
        training_data=census_train_data_numeric,
        training_labels=census_train_label_numeric,
        test_data=census_test_data_numeric,
        test_labels=census_test_label_numeric,
        epoch_count=epoch_count,
    )

    acc, fit_time, predict_time = review_model(
        census_net,
        census_train_data_numeric,
        census_train_label_numeric,
        census_test_data_numeric,
        census_test_label_numeric,
        census_net=True,
    )
    results.append(
        [
            "Neural Network",
            "Census",
            "network_learning_rate = 0.000001, layer_size = 16, epoch_count = 275",
            acc,
            fit_time,
            predict_time,
        ]
    )

    # MNIST
    (
        mn_train_data_unflattened,
        mn_train_one_hot_labels,
        mn_train_labels,
        mn_test_images_unflattened,
        mn_test_one_hot_labels,
        mn_test_labels,
    ) = get_mnist_data_labels_neural(flatten_images=False)

    use_cnn = True
    epoch_count = 100
    batch_size = 100
    mnist_net = MNISTNetCNN(
        use_cnn=use_cnn,
        epoch_count=epoch_count,
        batch_size=batch_size,
        train_data=mn_train_data_unflattened,
        train_one_hot_labels=mn_train_one_hot_labels,
        train_labels=mn_train_labels,
        cv_data=None,
        cv_labels=None,
        test_data=mn_test_images_unflattened,
        test_labels=mn_test_labels,
    )

    acc, fit_time, predict_time = review_model(
        mnist_net,
        mn_train_data_unflattened,
        mn_train_one_hot_labels,
        mn_test_images_unflattened,
        mn_test_labels,
        neural_net=True,
    )
    results.append(
        [
            "Neural Network",
            "MNIST",
            "use_cnn=True, epoch_count=100",
            acc,
            fit_time,
            predict_time,
        ]
    )

    # Boosting
    ################################################################
    # Census
    loss = "exponential"
    learning_rate = 1
    n_estimators = 200
    max_features = 5
    max_depth = 3

    census_boost = GradientBoostingClassifier(
        loss=loss,
        learning_rate=1,
        random_state=random_state,
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
    )

    acc, fit_time, predict_time = review_model(
        census_boost,
        census_train_data_numeric,
        census_train_label_numeric.ravel(),
        census_test_data_numeric,
        census_test_label_numeric,
    )
    results.append(
        [
            "Boosting ",
            "Census",
            "n_estimators=200, max_depth=3, max_features=5",
            acc,
            fit_time,
            predict_time,
        ]
    )

    # MNIST
    loss = "deviance"
    n_estimators = 25
    max_features = 38
    max_depth = 15

    mnist_boost = GradientBoostingClassifier(
        loss=loss,
        learning_rate=1,
        random_state=random_state,
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
    )

    acc, fit_time, predict_time = review_model(
        mnist_boost,
        mn_train_data,
        np.reshape(mn_train_labels.numpy(), (len(mn_train_labels), 1)).ravel(),
        mn_test_images,
        np.reshape(mn_test_labels.numpy(), (len(mn_test_labels), 1)),
    )
    results.append(
        [
            "Boosting ",
            "MNIST",
            "n_estimators=25, max_depth=15, max_features=38",
            acc,
            fit_time,
            predict_time,
        ]
    )

    # SVM
    ################################################################
    # Census
    kernel = "poly"
    census_svc = SVC(random_state=random_state, kernel=kernel)
    acc, fit_time, predict_time = review_model(
        census_svc,
        census_train_data_numeric,
        census_train_label_numeric.ravel(),
        census_test_data_numeric,
        census_test_label_numeric,
    )
    results.append(
        [
            "Support Vector Machine ",
            "Census",
            "kernel=poly",
            acc,
            fit_time,
            predict_time,
        ]
    )

    # MNIST
    kernel = "rbf"
    mnist_svc = SVC(random_state=random_state, kernel=kernel)
    acc, fit_time, predict_time = review_model(
        mnist_svc,
        mn_train_data,
        np.reshape(mn_train_labels.numpy(), (len(mn_train_labels), 1)).ravel(),
        mn_test_images,
        np.reshape(mn_test_labels.numpy(), (len(mn_test_labels), 1)),
    )
    results.append(
        [
            "Support Vector Machine ",
            "MNIST",
            "kernel=rbf",
            acc,
            fit_time,
            predict_time,
        ]
    )

    # KNN
    ################################################################
    # Census
    n_neighbors = 15
    weights = "uniform"
    metric = "manhattan"
    census_knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
    acc, fit_time, predict_time = review_model(
        census_knn,
        census_train_data_numeric,
        census_train_label_numeric.ravel(),
        census_test_data_numeric,
        census_test_label_numeric,
    )
    results.append(
        [
            "K Nearest Neighbors",
            "Census",
            "n_neighbors=15, weights='uniform',metric='manhattan'",
            acc,
            fit_time,
            predict_time,
        ]
    )

    # MNIST
    n_neighbors = 3
    weights = "distance"
    metric = "minkowski"
    mnist_knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
    acc, fit_time, predict_time = review_model(
        mnist_knn,
        mn_train_data,
        np.reshape(mn_train_labels.numpy(), (len(mn_train_labels), 1)).ravel(),
        mn_test_images,
        np.reshape(mn_test_labels.numpy(), (len(mn_test_labels), 1)),
    )
    results.append(
        [
            "K Nearest Neighbors",
            "MNIST",
            "n_neighbors=3, weights='distance',metric='minkowski'",
            acc,
            fit_time,
            predict_time,
        ]
    )
