#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q3.1a
        
        # predicted class
        y_hat = np.argmax(np.dot(self.W, x_i.T), axis = 0)
        
        # update parameters
        if y_hat != y_i:
            
            # update weights of true class
            self.W[y_i] += x_i 
            # update weights of predicted class
            self.W[y_hat] -= x_i
            
        return self.W        


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q3.1b
        
        # Label scores according to the model (n_classes,)
        scores = np.dot(self.W, x_i)
        
        # One-hot vector with the true label (n_classes,)
        y_one_hot = np.zeros(self.W.shape[0])
        y_one_hot[y_i] = 1
        
        # Softmax function.
        # This gives the label probabilities according to the model (n_classes,)
        label_probabilities = np.exp(scores) / np.sum(np.exp(scores))
        
        # SGD update. W is n_classes x n_features.
        self.W += learning_rate * (y_one_hot - label_probabilities)[:,None] * x_i.T
        
        return self.W


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers):
        # Initialize an MLP with a single hidden layer.
        
        self.n_classes = n_classes
        self.layers = layers

        # Initialize weights (W1, ..., W(layers+1))
        self.weights = []
        for i in range(layers + 1):
            if i == 0:
                self.weights.append(np.random.normal(0.1, 0.1, size = (n_features, hidden_size)))
            elif i == layers:
                self.weights.append(np.random.normal(0.1, 0.1, size = (hidden_size, n_classes)))
            else:
                self.weights.append(np.random.normal(0.1, 0.1, size = (hidden_size, hidden_size)))
        
        # Initialize biases (b1, ..., b(layers+1))
        self.biases = []
        for i in range(layers + 1):
            if i == 0:
                self.biases.append(np.zeros(hidden_size))
            elif i == layers:
                self.biases.append(np.zeros(n_classes))
            else:
                self.biases.append(np.zeros(hidden_size))   

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        hiddens = []
        for i in range(self.layers + 1):
            h = X if i == 0 else hiddens[i-1]
            z = np.dot(h, self.weights[i]) + self.biases[i]
            if i < self.layers:
                hiddens.append(np.maximum(z, 0))
        output = z
        yhat = np.argmax(output, axis = 1)
        return yhat      
       
    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        
        # stochastic gradient descent with batch size = 1
        for x_i, y_i in zip(X, y):
            
            # forward pass:
            hiddens = []
            for i in range(self.layers + 1):
                h = x_i if i == 0 else hiddens[i-1]
                z = np.dot(h, self.weights[i]) + self.biases[i]
                if i < self.layers:
                    hiddens.append(np.maximum(z,0))
            yhat = z
            
            # backward pass:
            
            # softmax activations
            probs = np.exp(yhat - np.max(yhat)) / np.sum(np.exp(yhat - np.max(yhat)))
            # one hot encoded vector of true class
            y_i_one_hot_vector = np.zeros(self.n_classes)
            y_i_one_hot_vector[y_i] = 1
            
            # dL/dz
            grad_z = probs - y_i_one_hot_vector # (n_classes,)
            
            grad_weights = []
            grad_biases = []
            
            for i in range(self.layers, -1, -1):
                
                h = x_i if i == 0 else hiddens[i-1]
                # compute gradient of parameters:
                grad_weights.append(np.dot(h[:,None], grad_z[:,None].T)) # dL/dw
                grad_biases.append(grad_z) # dL/db
        
                # compute gradient of hidden layer after activation function:
                grad_h = np.dot(grad_z, self.weights[i].T) # dL/dh
        
                # compute gradient of hidden layer before activation:
                # (h > 0) to get the gradient of the ReLU
                grad_z = grad_h * (h > 0) # dL/dz
                
            grad_weights.reverse()
            grad_biases.reverse()
        
            # updating the weights
            for i in range(self.layers + 1):
                self.weights[i] -= learning_rate * grad_weights[i]
                self.biases[i] -= learning_rate * grad_biases[i]

        return self.weights, self.biases


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))
        print('Accuracy (val): %.3f | Accuracy (test): %.3f' % (valid_accs[-1], test_accs[-1]))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
