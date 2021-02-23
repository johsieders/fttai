
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# X = Input 4 x 2
# Y = Output  4 x 1


class Learner(object):
    def __init__(self, X, Y, G, optimizer, loss):
        self.X = X
        self.Y = Y
        self.G = G
        self.optimizer = optimizer
        self.loss = loss

    def fit(self):
        cnt = 0
        history = []

        while cnt < 1001:
            currentLoss = self.loss(self.G(self.X), self.Y)
            if cnt % 100 == 0:
                history.append(currentLoss.item())
            currentLoss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            cnt += 1

        return history

    def result(self):
        return self.G(self.X)


def prepare21(lr):
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=True, dtype=torch.float32)
    Y = torch.tensor([[0], [1], [1], [1]], dtype=torch.float32)
    G = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

    d = {'0.weight': torch.tensor([[1., 1.]]),
         '0.bias': torch.tensor([0.0])}
    G.load_state_dict(d)

    optimizer = optim.SGD(G.parameters(), lr=lr)
    loss = nn.MSELoss()

    return Learner(X, Y, G, optimizer, loss)


def prepare21a(lr):
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=True, dtype=torch.float32)
    Y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
    G = nn.Linear(2, 1)

    # d = {'weight': torch.tensor([[1., 1.]]),
    #      'bias': torch.tensor([0.0])}
    # G.load_state_dict(d)
    # lr = 0.009
    optimizer = optim.SGD(G.parameters(), lr=lr)
    loss = nn.BCEWithLogitsLoss()

    return Learner(X, Y, G, optimizer, loss)


def prepare22(lr):
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=True, dtype=torch.float32)
    Y = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    G = nn.Linear(2, 2)
    #
    # d = {'weight': torch.tensor([[1., 1.], [1., 1.]]),
    #      'bias': torch.tensor([0., 0.])}
    # G.load_state_dict(d)

    optimizer = optim.SGD(G.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    return Learner(X, Y, G, optimizer, loss)


def plot(history):
    plt.plot(history, label='training loss')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('loss')
    plt.xlabel('iterations')


class TestTorch(unittest.TestCase):
    def testBool(self):
        # learn = prepare21(0.3)
        # learn = prepare21a(0.009)
        learn = prepare22(0.009)
        print(learn.result())
        history = learn.fit()
        print(learn.result())
        plot(history)