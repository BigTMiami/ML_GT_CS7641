from Assignment_1.prep_census_data import get_census_data_and_labels
from Assignment_1.mnist_data_prep import get_mnist_data_labels
from sklearn import tree
import matplotlib.pyplot as plt

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

# Train Data
decision_tree = tree.DecisionTreeClassifier(max_depth=3)
dt = decision_tree.fit(df_data_numeric, df_label_numeric)

# predict_results = dt.predict(df_test_data_numeric)
score = dt.score(df_test_data_numeric, df_test_label_numeric)
print(f"Census:{score}")

# info = tree.plot_tree(dt)
# plt.show()

train_images_flattened, train_labels, test_images_flattened, test_labels = get_mnist_data_labels()

# Train Data
decision_tree = tree.DecisionTreeClassifier(max_depth=9)
dt = decision_tree.fit(train_images_flattened, train_labels)

# predict_results = dt.predict(df_test_data_numeric)
score = dt.score(test_images_flattened, test_labels)
print(f"MNIST Image:{score}")
