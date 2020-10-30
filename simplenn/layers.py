import numpy as np


class Layer:
    """ Base class and interface for layers of a NeuralNet.

        All layers need `params` and `grads` for the NeuralNet to work.
        Layers may leave them empty, though.
    """
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Linear(Layer):
    """ Linear layer with output = inputs · w + b
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        # Linear back propagation. Quick mafs by Joel:
        #
        # if y = f(x) and x = a * b + c
        # then dy/da = f'(x) * b
        # and dy/db = f'(x) * a
        # and dy/dc = f'(x)
        #
        # if y = f(x) and x = a @ b + c
        # then dy/da = f'(x) @ b.T
        # and dy/db = a.T @ f'(x)
        # and dy/dc = f'(x)
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


class Activation(Layer):
    """ Activation layer with output = f(inputs)
    """
    def __init__(self, f, f_prime):
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs):
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad):
        # chain rule:
        #
        # if y = f(x) and x = g(z)
        # then dy/dz = f'(x) * g'(z)
        return self.f_prime(self.inputs) * grad


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    y = tanh(x)
    return 1 - y ** 2
