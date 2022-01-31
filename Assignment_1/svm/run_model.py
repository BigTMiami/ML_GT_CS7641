from sklearn.model_selection import cross_val_score


def train_with_cv_mulit(kernel, train_data, train_labels, test_data, test_labels, use_cv, cv_count, model):
    print(f"Starting {kernel}")
    if use_cv:
        cv_score = cross_val_score(model, train_data, train_labels, cv=cv_count).mean()
    else:
        cv_score = 0.0

    model.fit(train_data, train_labels)
    train_score = model.score(train_data, train_labels)
    test_score = model.score(test_data, test_labels)

    return kernel, train_score, cv_score, test_score
