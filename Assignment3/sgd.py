#################################
# Your name: Ram Elgov (206867517)
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd_drive.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    data_count = data.shape[0]
    data_dimension = data.shape[1]
    w = np.zeros(data_dimension)
    for t in range(1, T + 1):
        i = np.random.randint(0, data_count, 1)[0]
        eta_t = eta_0 / t  # learning rate function
        x_i = data[i]
        y_i = labels[i]
        if np.dot(y_i * w, x_i) < 1:
            w = (1 - eta_t) * w + eta_t * C * y_i * x_i
        else:
            w = (1 - eta_t) * w
    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    len_data = data.shape[0]
    dim_data = data.shape[1]
    w = np.zeros(dim_data)
    for t in range(1, T + 1):
        i = np.random.randint(0, len_data, 1)[0]
        eta_t = eta_0 / t  # learning rate function
        x_i = data[i]
        y_i = labels[i]
        # Compute the gradient of log loss
        exp_term = np.exp(-y_i * np.dot(w, x_i))
        gradient = -(1 / (1 + exp_term)) * exp_term * y_i * x_i
        # Update the weights
        w -= eta_t * gradient
    return w


#################################

# Place for additional code

#################################

def average_accuracy_eta_0_hinge(runs=10):
    """ Plots the average accuracy of SGD with Hinge loss as a function of eta_0."""
    etas = [10 ** i for i in range(-5, 6)]
    accuracies = []
    for eta_0 in etas:
        sum_accuracy = 0
        for i in range(runs):
            w = SGD_hinge(train_data, train_labels, 1, eta_0, 1000)
            accuracy = calculate_accuracy(validation_data, validation_labels, w)
            sum_accuracy += accuracy
        accuracies.append(sum_accuracy / runs)
    # plot
    plt.title("average accuracy of SGD with Hinge loss as a function of eta_0")
    plt.xlabel('eta_0')
    plt.ylabel('average accuracy')
    plt.xscale('log')
    plt.plot(etas, accuracies)
    plt.show()

    return etas[np.argmax(accuracies)]


def calculate_accuracy(data, labels, w):
    """ Calculates the accuracy of the given data and labels with the given weights."""
    data_count = data.shape[0]
    correct_predictions = 0
    for i in range(data_count):
        x_i = data[i]
        y_i = labels[i]
        y_predict = 1 if np.dot(x_i, w) >= 0 else -1
        correct_predictions += 1 if y_i == y_predict else 0
    return correct_predictions / data_count


def average_accuracy_C(eta_0, runs=10):
    """ Plots the average accuracy of SGD with Hinge loss as a function of C."""
    C_candidates = [10 ** i for i in range(-5, 6)]
    accuracies = []
    for c in C_candidates:
        sum_accuracy = 0
        for i in range(runs):
            w = SGD_hinge(train_data, train_labels, c, eta_0, 1000)
            accuracy = calculate_accuracy(validation_data, validation_labels, w)
            sum_accuracy += accuracy
        accuracies.append(sum_accuracy / runs)
    # plot
    plt.title("average accuracy of SGD with Hinge loss as a function of C")
    plt.xlabel('C')
    plt.ylabel('average accuracy')
    plt.xscale('log')
    plt.plot(C_candidates, accuracies)
    plt.show()

    return C_candidates[np.argmax(accuracies)]


def plot_w_image(eta_0, c=0, loss="hinge"):
    """ Plots the image of the classifier w."""
    if loss == "hinge":
        w = SGD_hinge(train_data, train_labels, c, eta_0, 20000)
    else:
        w = SGD_log(train_data, train_labels, eta_0, 20000)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()


def accuracy_best_classifier(eta_0, c=0, loss="hinge"):
    """ Returns the accuracy of the best classifier on the test data."""
    if loss == "hinge":
        w = SGD_hinge(train_data, train_labels, c, eta_0, 20000)
    else:
        w = SGD_log(train_data, train_labels, eta_0, 20000)
    return calculate_accuracy(test_data, test_labels, w)


def average_accuracy_eta_0_log(runs=10):
    """ Plots the average accuracy of SGD with Log loss as a function of eta_0."""
    etas = [10 ** i for i in range(-5, 6)]
    accuracies = []
    for eta_0 in etas:
        sum_accuracy = 0
        for i in range(runs):
            w = SGD_log(train_data, train_labels, eta_0, 1000)
            accuracy = calculate_accuracy(validation_data, validation_labels, w)
            sum_accuracy += accuracy
        accuracies.append(sum_accuracy / runs)
    # plot
    plt.title("average accuracy of SGD with Log loss as a function of eta_0")
    plt.xlabel('eta_0')
    plt.ylabel('average accuracy')
    plt.xscale('log')
    plt.plot(etas, accuracies)
    plt.show()

    return etas[np.argmax(accuracies)]


def plot_norm_w(data, labels, eta_0, T):
    """ Plots the norm of w as a function of iteration number."""
    len_data = data.shape[0]
    dim_data = data.shape[1]
    w = np.zeros(dim_data)
    norm_values = []
    for t in range(1, T + 1):
        i = np.random.randint(0, len_data, 1)[0]
        eta_t = eta_0 / t  # learning rate function
        x_i = data[i]
        y_i = labels[i]
        exp_term = np.exp(-y_i * np.dot(w, x_i))
        gradient = -(1 / (1 + exp_term)) * exp_term * y_i * x_i
        w -= eta_t * gradient
        norm = np.linalg.norm(w)  # Calculate the norm of w
        norm_values.append(norm)

    # Plot norm values as a function of iteration
    plt.plot(range(1, T + 1), norm_values)
    plt.title("Norm of w as a function of iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Norm of w")
    plt.show()


def run():
    """ Runs the program."""
    # Q1
    eta_0 = average_accuracy_eta_0_hinge()
    print("Question: 1")
    print("____________")
    print("optimal eta_0: ", eta_0)
    c = average_accuracy_C(eta_0)
    print("optimal C: ", c)
    plot_w_image(eta_0, c)
    print("accuracy of the best classifier: ", accuracy_best_classifier(eta_0, c))
    # Q2
    print("Question: 2")
    print("____________")
    eta_0 = average_accuracy_eta_0_log()
    print("optimal eta_0: ", eta_0)
    plot_w_image(eta_0, loss="log")
    print("accuracy of the best classifier: ", accuracy_best_classifier(eta_0, loss="log"))
    plot_norm_w(train_data, train_labels, eta_0, 20000)


if __name__ == "__main__":
    # Driver Code
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    run()
