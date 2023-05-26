import mxnet as mx
import numpy as np
import os
from mxnet.gluon.nn import Sequential
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.optim import Adam


class MNISTNet(Sequential):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.add(mx.gluon.nn.Dense(784, 512))
        self.add(mx.gluon.nn.Activation('relu'))
        self.add(mx.gluon.nn.Dropout(0.2))
        self.add(mx.gluon.nn.Dense(512, 10))
        self.add(mx.gluon.nn.SoftmaxOutput())


def load_data(data_dir):
    path = os.path.join(data_dir, 'mnist.npz')
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test


def train(model, train_data, train_labels, test_data, test_labels, epochs, batch_size, lr):
    # Create an optimizer
    optimizer = Adam(learning_rate=lr)

    # Create a loss function
    loss_fn = SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        # Train for one epoch
        for batch_idx, (data, label) in enumerate(train_data):
            # Get the predictions
            predictions = model(data)

            # Calculate the loss
            loss = loss_fn(predictions, label)

            # Backpropagate the loss
            optimizer.backward(loss)

            # Update the parameters
            optimizer.step()

            # Print the loss
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_data),
                    100. * batch_idx / len(train_data), loss.asscalar()))

        # Evaluate on the test set
        test_loss = 0
        correct = 0
        with mx.nd.stop.gradient:
            for data, label in test_data:
                # Get the predictions
                predictions = model(data)

                # Calculate the loss
                loss = loss_fn(predictions, label)

                # Count the number of correct predictions
                correct += mx.nd.sum(mx.argmax(predictions, axis=1) == label)

        # Calculate the accuracy
        test_accuracy = correct / len(test_data)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_data),
            100. * correct / len(test_data)))


def main():
    # Get the data directory
    data_dir = os.getenv('DATA_DIR', '/platform/data')

    # Get the scratch directory
    scratch_dir = os.getenv('SCRATCH_DIR', '/platform/scratch')

    # Get the model directory
    model_dir = os.getenv('MODEL_DIR', '/platform/model')

    # Get the number of epochs
    epochs = int(os.getenv('MAX_EPOCHS', '5'))

    # Get the batch size
    batch_size = int(os.getenv('BATCH_SIZE', '128'))

    # Get the learning rate
    lr = float(os.getenv('LEARNING_RATE', '0.001'))

    # Load the data
    x_train, y_train, x_test, y_test = load_data(data_dir)

    # Create the model
    model = MNISTNet()

    # Train the model
    train(model, x_train, y_train, x_test, y_test, epochs, batch_size, lr)

    # Save the model
    model.save(os.path.join(scratch_dir, model_name))


if __name__ == '__main__':
    main()

