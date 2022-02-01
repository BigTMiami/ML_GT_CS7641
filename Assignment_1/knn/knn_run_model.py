from sklearn.model_selection import cross_val_score


def train_with_cv_multi(
    n_neighbors, weights, metric, train_data, train_labels, test_data, test_labels, use_cv, cv_count, model
):
    print(f"Starting {n_neighbors}, {weights}, {metric}")
    if use_cv:
        cv_score = cross_val_score(model, train_data, train_labels, cv=cv_count).mean()
    else:
        cv_score = 0.0

    model.fit(train_data, train_labels)
    # train_score = model.score(train_data, train_labels)
    train_score = 0
    test_score = model.score(test_data, test_labels)
    print("\a")
    print(
        f"{n_neighbors}, {weights}, {metric}: train:{train_score*100.0:6.3f}, cv:{cv_score*100.0:6.3f}, test:{test_score*100.0:6.3f} "
    )

    return n_neighbors, weights, metric, train_score, cv_score, test_score
