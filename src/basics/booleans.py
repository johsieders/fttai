import torch
from torch import tensor


# Native boolean functions cannot be applied to tensors.
# But these functions can.


def xor_(a, b):
    """
    :param a: 0 or 1 as int or float
    :param b: 0 or 1 as int or float
    :return: xor(a, b)
    """
    return a + b - 2 * a * b


def and_(a, b):
    """
    :param a: 0 or 1 as int or float
    :param b: 0 or 1 as int or float
    :return: and(a, b)
    """
    return a * b


def or_(a, b):
    """
    :param a: 0 or 1 as int or float
    :param b: 0 or 1 as int or float
    :return: or(a, b)
    """
    return a + b - a * b


if __name__ == '__main__':
    x = (0, 0, 0, 1, 1, 0, 1, 1)
    X = tensor(x, dtype=torch.float32).view(4, 2)

    print(and_(X[:, 0], X[:, 1]))
    print(or_(X[:, 0], X[:, 1]))
    print(xor_(X[:, 0], X[:, 1]))
