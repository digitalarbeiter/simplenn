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


def train(net, loss, optimizer, inputs, targets, n_epochs=5000):
    """ Train the neural net with the given inputs/targets, using the
        given loss function and optimizer, over n_epochs.
    """
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in batch(inputs, targets):
            predicted = net.forward(batch_inputs)
            batch_loss, grad = loss(predicted, batch_targets)
            epoch_loss += batch_loss
            net.backward(grad)
            optimizer(net)
        print(epoch, epoch_loss)
