from Assignment_1.prep_census_data import get_census_data_and_labels
from Assignment_1.mnist_data_prep import get_mnist_data_labels
from sklearn import tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os


def title_to_filename(title, classifier_type):
    return f"Assignment_1/document/figures/{title.replace(' ','_')}_{classifier_type}.png"


def get_decision_tree_pruning_scores(train_data, train_lables, test_data, test_labels, max_depth):
    # Train Data
    decision_tree = tree.DecisionTreeClassifier(max_depth=max_depth)

    # Get Alpha Values for pruning curve
    ccp_path = decision_tree.cost_complexity_pruning_path(df_data_numeric, df_label_numeric)
    print(f"{len(ccp_path['ccp_alphas'])}")

    alpha_scores = []
    for alpha in ccp_path["ccp_alphas"]:
        dt = tree.DecisionTreeClassifier(max_depth=max_depth, ccp_alpha=alpha)
        dt.fit(df_data_numeric, df_label_numeric)
        alpha_scores.append(dt.score(df_test_data_numeric, df_test_label_numeric))

    for alpha, score in zip(ccp_path["ccp_alphas"], alpha_scores):
        print(f"{alpha:0.5f}:{score:0.4f}")


def get_decision_tree_pruning_curve(
    train_data, train_labels, test_data, test_labels, criterion="gini", random_state=0
):
    decision_tree = tree.DecisionTreeClassifier(criterion=criterion, random_state=random_state)
    ccp_path = decision_tree.cost_complexity_pruning_path(train_data, train_labels)
    all_alphas = ccp_path["ccp_alphas"]
    # Sample Alphas
    alpha_count = len(all_alphas)
    alpha_increment = int(alpha_count / 50)
    alphas = [all_alphas[i] for i in range(0, alpha_count, alpha_increment)]
    alpha_scores = []
    alpha_node_count = []
    alpha_max_depth = []

    print(f"{'alpha':9} {'score':9} {'N_cnt':5} {'M_dpt':5} ")
    for alpha in alphas:
        dt = tree.DecisionTreeClassifier(ccp_alpha=alpha, random_state=random_state)
        dt.fit(train_data, train_labels)
        score = dt.score(test_data, test_labels)
        node_count = dt.tree_.node_count
        max_depth = dt.tree_.max_depth
        alpha_scores.append(score)
        alpha_node_count.append(node_count)
        alpha_max_depth.append(max_depth)
        print(f"{alpha:0.7f} {score:0.5f} {node_count:5} {max_depth:5} ")

    return alphas, alpha_scores, alpha_node_count, alpha_max_depth


def get_decision_tree_pruning_curve_with_cross_validation(
    train_data, train_labels, test_data, test_labels, criterion="gini", cross_validations=5, random_state=0
):
    decision_tree = tree.DecisionTreeClassifier(criterion=criterion, random_state=random_state)
    ccp_path = decision_tree.cost_complexity_pruning_path(train_data, train_labels)
    all_alphas = ccp_path["ccp_alphas"]
    # Sample Alphas
    alpha_count = len(all_alphas)
    alpha_increment = int(alpha_count / 50)
    alphas = [all_alphas[i] for i in range(0, alpha_count, alpha_increment)]
    alpha_test_scores = []
    alpha_train_scores = []
    alpha_node_count = []
    alpha_max_depth = []

    cross_validation_scores = []

    print(f"{'alpha':9} {'score':9} {'N_cnt':5} {'M_dpt':5} {'Train':7} {'CV':7}")
    for alpha in alphas:
        dt = tree.DecisionTreeClassifier(ccp_alpha=alpha, random_state=random_state)
        dt.fit(train_data, train_labels)
        score = dt.score(test_data, test_labels)
        train_score = dt.score(train_data, train_labels)
        alpha_train_scores.append(train_score)

        node_count = dt.tree_.node_count
        max_depth = dt.tree_.max_depth
        alpha_test_scores.append(score)
        alpha_node_count.append(node_count)
        alpha_max_depth.append(max_depth)

        dt = tree.DecisionTreeClassifier(ccp_alpha=alpha, random_state=random_state)
        cv_score = cross_val_score(dt, train_data, train_labels, cv=cross_validations).mean()
        cross_validation_scores.append(cv_score)
        print(f"{alpha:0.7f} {score:0.5f} {node_count:5} {max_depth:5} {train_score:0.5f} {cv_score:0.5f}")

    return alphas, alpha_test_scores, alpha_node_count, alpha_max_depth, alpha_train_scores, cross_validation_scores


def plot_alpha_curve(alphas, alpha_scores, alpha_node_count, alpha_max_depth, title, xscale="linear", nodes_to_trim=3):
    fig, ax = plt.subplots(2, figsize=(4, 5))
    fig.suptitle(title, fontsize=16)
    ax[0].set_title("Accuracy")
    ax[0].plot(alphas[nodes_to_trim:], alpha_scores[nodes_to_trim:])
    ax[0].set_xlabel("Alpha")
    ax[0].set_xscale(xscale)
    ax[0].invert_xaxis()
    ax[0].set_ylabel("Test Score")
    ax[0].autoscale_view("tight")

    ax[1].set_title("Tree Size")
    ax[1].plot(alphas[nodes_to_trim:], alpha_max_depth[nodes_to_trim:], color="blue")
    ax[1].set_xlabel("Alpha")
    ax[1].set_ylabel("Max Depth", color="blue")
    ax[1].tick_params(axis="y", labelcolor="blue")
    ax[1].set_xscale(xscale)
    ax[1].invert_xaxis()
    ax[1].autoscale_view("tight")
    ax2 = ax[1].twinx()
    ax2.plot(alphas[nodes_to_trim:], alpha_node_count[nodes_to_trim:], color="red")
    ax2.set_ylabel("Node Count", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    filename = title_to_filename(title, "decision_tree")
    if os.path.exists(filename):
        os.remove(filename)
    fig.tight_layout()
    plt.savefig(fname=filename, bbox_inches="tight")


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


alphas, alpha_scores, alpha_node_count, alpha_max_depth = get_decision_tree_pruning_curve(
    df_data_numeric, df_label_numeric, df_test_data_numeric, df_test_label_numeric
)

(
    alphas,
    alpha_scores,
    alpha_node_count,
    alpha_max_depth,
    alpha_train_scores,
    cross_validation_scores,
) = get_decision_tree_pruning_curve_with_cross_validation(
    df_data_numeric, df_label_numeric, df_test_data_numeric, df_test_label_numeric
)

plot_alpha_curve(alphas, alpha_scores, alpha_node_count, alpha_max_depth, "Census Data", xscale="log", nodes_to_trim=3)

print(f"Max Cencus Accuracy:{max(alpha_scores)}")

# Image Data
train_images_flattened, train_labels, test_images_flattened, test_labels = get_mnist_data_labels()

alphas_image, alpha_scores_image, alpha_node_count_image, alpha_max_depth_image = get_decision_tree_pruning_curve(
    train_images_flattened, train_labels, test_images_flattened, test_labels, criterion="entropy"
)

(
    alphas_image,
    alpha_scores_image,
    alpha_node_count_image,
    alpha_max_depth_image,
    alpha_train_scores_image,
    cross_validation_scores_image,
) = get_decision_tree_pruning_curve_with_cross_validation(
    train_images_flattened, train_labels, test_images_flattened, test_labels
)

plot_alpha_curve(
    alphas_image,
    alpha_scores_image,
    alpha_node_count_image,
    alpha_max_depth_image,
    "MNIST Images",
    xscale="log",
    nodes_to_trim=0,
)

print(f"Max MNIST Accuracy:{max(alpha_scores_image)}")
