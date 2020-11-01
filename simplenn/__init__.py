""" Simple Neural Network.
"""
import numpy as np


def batch(inputs, targets, batch_size=32, shuffle=True):
    """ Go through inputs and targets and return (yield, actually) batches
        of given batch size.
    """
    starts = np.arange(0, len(inputs), batch_size)
    if shuffle:
        np.random.shuffle(starts)
    for start in starts:
        end = start + batch_size
        batch_inputs = inputs[start:end]
        batch_targets = targets[start:end]
        yield batch_inputs, batch_targets


class NeuralNet:
    """ A Neural Network is a bunch of layers with methods to forward
        and backward propagate through these layers.
    """
    def __init__(self, *layers):
        """ Initialize Neural Network with given layers.
        """
        self.layers = list(layers)
        print("layers: ", len(self.layers))

    def predict(self, inputs):
        """ Predict targets for inputs.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward_propagate(self, grad):
        """ Backward propagate gradient.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train(self, loss, optimizer, inputs, targets, n_epochs=5000):
        """ Train the neural net with the given inputs/targets, using the
            given loss function and optimizer, over n_epochs.
        """
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch_inputs, batch_targets in batch(inputs, targets):
                predicted = self.predict(batch_inputs)
                batch_loss, grad = loss(predicted, batch_targets)
                epoch_loss += batch_loss
                self.backward_propagate(grad)
                optimizer(self)
            print(epoch, epoch_loss)

    def params_and_grads(self):
        """ All parameters and their gradients, of all layers.
        """
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad


def sgd_optimizer(learning_rate=0.01):
    """ Stochastic Gradient Descent optimizer, with given learning rate.
    """
    def _step(net):
        for param, grad in net.params_and_grads():
            param -= learning_rate * grad
    return _step


def tse_loss(predicted, actual):
    """ Total Squared Error loss function, returns error and gradient.
    """
    return (
        np.sum((predicted - actual) ** 2),
        2 * (predicted - actual),
    )


# TODO: implement Mean Squared Error mse(predicted, actual)
