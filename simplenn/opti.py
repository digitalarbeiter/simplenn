import numpy as np


def tse(predicted, actual):
    """ Total Squared Error loss function, returns error and gradient.
    """
    return (
        np.sum((predicted - actual) ** 2),
        2 * (predicted - actual),
    )


# TODO: implement Mean Squared Error mse(predicted, actual)


def sgd(lr=0.01):
    """ Stochastic Gradient Descent optimizer, with given learning rate.
    """
    def _step(net):
        for param, grad in net.params_and_grads():
            param -= lr * grad
    return _step
