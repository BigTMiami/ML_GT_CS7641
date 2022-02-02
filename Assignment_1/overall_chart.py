from Assignment_1.overall_results import results
import matplotlib.pyplot as plt
import os
import numpy as np


def title_to_filename(title, classifier_type):
    safe_title = title.replace(" ", "_")
    safe_title = safe_title.replace(":", "_")
    safe_title = safe_title.replace(",", "_")
    return f"Assignment_1/document/figures/working/{safe_title}_{classifier_type}.png"


def chart_overall_bar(results, y_col, y_label, title, subtitle):
    columns = ["algorithm", "dataset", "settings", "acc", "fit_time", "predict_time"]
    y_col_index = columns.index(y_col)

    for algorithm, dataset, settings, acc, fit_time, predict_time in results:
        print(
            f"{algorithm:>25}  {dataset:>15}:  acc: {acc:6.3f}  fit_time: {fit_time:8.3f} predict_time: {predict_time:8.3f}"
        )

    x_algorithms = [r[0] for r in results if r[1] == "Census"]
    x_axis = np.arange(len(x_algorithms))

    y_census = [r[y_col_index] for r in results if r[1] == "Census"]
    y_mnist = [r[y_col_index] for r in results if r[1] == "MNIST"]

    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=16)
    ax.set_title(subtitle)

    plt.bar(x_axis - 0.2, y_census, 0.4, label="Census")
    plt.bar(x_axis + 0.2, y_mnist, 0.4, label="MNIST")
    ax.set_xlabel("Algorithms")
    plt.xticks(x_axis, x_algorithms, rotation=45)
    ax.set_ylabel(y_label)
    y_min = min(y_census + y_mnist)
    y_max = max(y_census + y_mnist)
    y_adj = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - y_adj, y_max + y_adj)
    plt.legend()

    filename = title_to_filename(title + subtitle, "")
    if os.path.exists(filename):
        os.remove(filename)
    fig.tight_layout()
    plt.savefig(fname=filename, bbox_inches="tight")


if __name__ == "main":
    chart_overall_bar(results, "acc", "Accuracy %", "Overall Algorithm Review", "Comparison of Accuracy")
    chart_overall_bar(
        results, "fit_time", "Learning Time (s)", "Overall Algorithm Review", "Comparison of Learning Time"
    )
    chart_overall_bar(
        results, "predict_time", "Prediction Time (s)", "Overall Algorithm Review", "Comparison of Prediction Time"
    )
