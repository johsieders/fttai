# neural metworks support: torch, sklearn metrics
import matplotlib.pyplot as plt
# utilities for display
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import Tensor


def showProgress(protocol):
    plt.figure()
    plt.plot(protocol[1], label='training loss')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.show()


def showConfusionMatrix(labels, predictions: Tensor, names: list) -> None:
    """
    @param labels: tensor of labels
    @param predictions: tensor of prediction
    @param names: names of categories, e.g. ['correct', 'incorrect']
    @return: None
    The confusion matrix is a K x K matrix with K = number of categories
    """

    cm = confusion_matrix(labels, predictions)
    vmax = cm.max()  # number of categories
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=True,
                xticklabels=names, yticklabels=names, vmin=0, vmax=vmax, cmap="YlGnBu")
    plt.xlabel = ('true label')
    plt.ylabel = ('predicted label')
    plt.show()


def showMetrics(labels, predictions: Tensor) -> None:
    """"
    @param labels: tensor of labels
    @param predictions: tensor of prediction
    @return: None
    This functions prints accuracy, precision, recall and f1
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    print(f'\naccuracy  = {accuracy:.3f}\n'
          f'precision = {precision:.3f}\n'
          f'recall    = {recall:.3f}\n'
          f'f1        = {f1:.3f}')


if __name__ == '__main__':
    # Playing with Confusion

    labels = torch.randint(5, (100, 1))  # 100 random ints in [0, 5)
    cat_names = ['a', 'b', 'c', 'd', 'e']
    no_errors = labels.clone()

    some_errors = no_errors.clone()
    some_errors[range(0, 100, 10)] = (some_errors[range(0, 100, 10)] + 1) % 5

    all_wrong = no_errors.clone()
    all_wrong[:] = (all_wrong[:] + 2) % 5

    showConfusionMatrix(labels, no_errors, cat_names)
    showConfusionMatrix(labels, some_errors, cat_names)
    showConfusionMatrix(labels, all_wrong, cat_names)

    # Playing with Metrics

    labels = torch.randint(2, (100, 1))  # 100 random ints in [0, 2)
    no_errors = labels.clone()

    some_errors = no_errors.clone()
    some_errors[range(0, 100, 10)] = 1 - some_errors[range(0, 100, 10)]

    # all_wrong = no_errors.clone()
    all_wrong = 1 - no_errors.clone()

    showMetrics(labels, no_errors)
    showMetrics(labels, some_errors)
    showMetrics(labels, all_wrong)
