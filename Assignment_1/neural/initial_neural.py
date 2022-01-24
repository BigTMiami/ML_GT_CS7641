from Assignment_1.neural.dqn_agent import DQNAgent
from Assignment_1.prep_census_data import get_census_data_and_labels

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

dqn = DQNAgent(training_data=df_data_numeric.to_numpy(), training_labels=df_label_numeric.to_numpy())

dqn.test()

dqn.train()

dqn.training_labels_tensor
