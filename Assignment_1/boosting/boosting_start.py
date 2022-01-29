from sklearn.ensemble import GradientBoostingClassifier
from Assignment_1.prep_census_data import get_census_data_and_labels
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


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

train_data_one_hot = pd.get_dummies(df_train_data).to_numpy()
test_data_one_hot = pd.get_dummies(df_test_data).to_numpy()

train_data_one_hot.shape
test_data_one_hot.shape

scaler = MinMaxScaler()
train_data_one_hot = scaler.fit_transform(train_data_one_hot)
test_data_one_hot = scaler.transform(test_data_one_hot)

train_data_one_hot.shape
df_train_data.shape

gb_clf = GradientBoostingClassifier(
    n_estimators=100, loss="exponential", learning_rate=1, max_features=2, max_depth=3, random_state=0
)
gb = gb_clf.fit(np_train_data_numeric, np_train_label_numeric.ravel())
gb_clf.score(np_train_data_numeric, np_train_label_numeric)
gb_clf.score(np_test_data_numeric, np_test_label_numeric)

gb_clf.estimators_

gb_clf.classes_
gb_clf.n_features_in_
gb_clf.feature_importances_
