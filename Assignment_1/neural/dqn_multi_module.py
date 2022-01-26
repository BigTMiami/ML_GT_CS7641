def run_training(split_count, dqn, cv_train_indexes, cv_test_indexes, show_details):
    # split_count, dqn, cv_train_indexes, cv_test_indexes, show_details = training_set
    epoch_values = dqn.train(
        show_details=show_details, cv_train_indexes=cv_train_indexes, cv_test_indexes=cv_test_indexes
    )

    return split_count, epoch_values
