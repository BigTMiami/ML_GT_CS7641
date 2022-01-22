from Assignment_1.prep_census_data import get_census_data_and_labels
from Assignment_1.mnist_data_prep import get_mnist_data_labels
from sklearn import preprocessing, tree
import matplotlib.pyplot as plt


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


def get_decision_tree_pruning_curve(train_data, train_labels, test_data, test_labels):

    decision_tree = tree.DecisionTreeClassifier()
    ccp_path = decision_tree.cost_complexity_pruning_path(train_data, train_labels)
    all_alphas = ccp_path["ccp_alphas"]
    # Sample Alphas
    alpha_count = len(all_alphas)
    alpha_increment = int(alpha_count / 50)
    alphas = [all_alphas[i] for i in range(0, alpha_count, alpha_increment)]
    alpha_scores = []
    alpha_node_count = []
    alpha_max_depth = []

    max_score = 0
    declining_score_count = 0
    max_declining = 15
    print(f"{'alpha':9} {'score':9} {'N_cnt':5} {'M_dpt':5} {'Decline':7}")
    for index, alpha in enumerate(alphas):
        dt = tree.DecisionTreeClassifier(ccp_alpha=alpha)
        dt.fit(train_data, train_labels)
        score = dt.score(test_data, test_labels)
        node_count = dt.tree_.node_count
        max_depth = dt.tree_.max_depth
        alpha_scores.append(score)
        alpha_node_count.append(node_count)
        alpha_max_depth.append(max_depth)
        print(f"{alpha:0.7f} {score:0.5f} {node_count:5} {max_depth:5} {declining_score_count:7}")

    return alphas, alpha_scores, alpha_node_count, alpha_max_depth


def plot_alpha_curve(alphas, alpha_scores, alpha_node_count, alpha_max_depth):
    start_node = 3
    fig, ax = plt.subplots(2, figsize=(3, 6))
    fig.suptitle("Decision Tree", fontsize=16)
    ax[0].set_title("Alpha vs Test Score")
    ax[0].plot(alphas[start_node:], alpha_scores[start_node:], color="blue")
    ax[0].set_xlabel("Alpha")
    ax[0].set_xscale("log")
    ax[0].set_ylabel("Test Score", color="blue")
    ax[0].tick_params(axis="y", labelcolor="blue")

    ax[1].set_title("Max Depth and Node Count")
    ax[1].plot(alphas[start_node:], alpha_max_depth[start_node:])
    ax[1].set_xlabel("Alpha")
    ax[1].set_ylabel("Max Depth")
    ax[1].set_xscale("log")
    ax2 = ax[1].twinx()
    ax2.plot(alphas[start_node:], alpha_node_count[start_node:], color="red")
    ax2.set_ylabel("Node Count", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    fig.tight_layout()
    plt.show()


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

plot_alpha_curve(alphas, alpha_scores, alpha_node_count, alpha_max_depth)

# Image Data
train_images_flattened, train_labels, test_images_flattened, test_labels = get_mnist_data_labels()

alphas_image, alpha_scores_image, alpha_node_count_image, alpha_max_depth_image = get_decision_tree_pruning_curve(
    train_images_flattened, train_labels, test_images_flattened, test_labels
)

plot_alpha_curve(alphas_image, alpha_scores_image, alpha_node_count_image, alpha_max_depth_image)
