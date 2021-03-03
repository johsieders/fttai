# Johannes Siedersleben
# QAware GmbH, Munich
# 28.2.2021

import torch
from torch import Tensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# utilities for display
import seaborn as sns
import matplotlib.pyplot as plt


def showProgress(protocol: list) -> None:
    """
    :param protocol: list of tuples (timestamp, loss)
    :return: None
    """
    losses = [protocol[i][1] for i in range(len(protocol)) if type(protocol[i][1]) is float]
    plt.figure()
    plt.plot(losses, label='training loss')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('loss')
    plt.xlabel('iterations')


def showMetrics(labels, predictions: Tensor) -> None:
    """"
    @param labels: tensor of labels
    @param predictions: tensor of prediction
    @return: None
    This functions prints accuracy, precision, recall and f1
    """
    # labels = torch.tensor(labels, device=torch.device('cpu'))
    # predictions = torch.tensor(predictions, device=torch.device('cpu'))
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    print(f'\naccuracy  = {accuracy:.3f}\n'
          f'precision = {precision:.3f}\n'
          f'recall    = {recall:.3f}\n'
          f'f1        = {f1:.3f}')


def showConfusionMatrix(labels, predictions: Tensor, names: list) -> None:
    """
    @param labels: tensor of labels
    @param predictions: tensor of prediction
    @param names: names of categories, e.g. ['correct', 'incorrect']
    @return: None
    The confusion matrix is a K x K matrix with K = number of categories
    """

    cm = confusion_matrix(labels, predictions)
    vmax = cm.max()   # number of categories
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=True,
                xticklabels=names, yticklabels=names, vmin=0, vmax=vmax, cmap="YlGnBu")
    plt.xlabel = ('true label')
    plt.ylabel = ('predicted label')
    plt.show()

