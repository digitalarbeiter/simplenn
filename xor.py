#!/usr/bin/env python3
# coding: utf-8

import numpy as np

from simplenn.layers import Linear, Activation, tanh, tanh_prime
from simplenn.nn import NeuralNet
from simplenn.opti import sgd, tse
from simplenn.train import train


if __name__ == "__main__":
    inputs = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    targets = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ])
    # XOR can't be learned with a simple linear model. Comment out the
    # Tanh and second Linear layer to see for yourself.
    net = NeuralNet([
        Linear(input_size=2, output_size=2),
        Activation(tanh, tanh_prime),
        Linear(input_size=2, output_size=2)
    ])
    train(
        net, loss=tse, optimizer=sgd(),
        inputs=inputs, targets=targets,
        n_epochs=5000,
    )
    print("x, predicted, y")
    for x, y in zip(inputs, targets):
        predicted = net.forward(x)
        print(x, predicted, y)
