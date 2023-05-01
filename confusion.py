import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def get_cm(label_matrix, prediction_matrix):
    L_np = label_matrix.cpu().numpy().flatten()
    P_np = prediction_matrix.cpu().numpy().flatten()

    cm = confusion_matrix(L_np, P_np)
    p_cm = np.zeros_like(cm, dtype=float)

    for i in range(len(p_cm)):
        for j in range(len(p_cm[0])):
            p_cm[i][j] = cm[i][j] / sum(cm[i])

    return p_cm


def plot_cm(matrix, save_pdf=False):
    # Set up the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a heatmap using Seaborn
    sns.heatmap(matrix, annot=True, cmap="coolwarm", cbar=True, ax=ax, vmin=0, vmax=1, square=True, fmt=".2f",
                linewidths=1, cbar_kws={"shrink": 0.5}, annot_kws={"size": 20})

    # Set plot title and axis labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Change class labels to "Background" and "Nerve"
    class_labels = ["Background", "Nerve"]
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xticks(np.arange(len(class_labels)) + 0.5)
    ax.set_yticks(np.arange(len(class_labels)) + 0.5)

    # Display the plot
    if save_pdf:
        plt.savefig("confusion_matrix.pdf", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':

    # Example get_cm
    L = torch.randint(0, 2, size=(20, 20))
    P = torch.randint(0, 2, size=(20, 20))
    cm_matrix = get_cm(L, P)

    # Example plot_cm
    M = np.array([[0.98, 0.02],
                  [0.10, 0.90]])
    plot_cm(M, save_pdf=True)
