import backprop_data
import backprop_network
import numpy as np
import matplotlib.pyplot as plt


def plot(learning_rates, y_data, y_label, title):
    for index, rate in enumerate(learning_rates):
        plt.plot(np.arange(30), y_data[index], label="rate = {}".format(rate), marker='.')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


# Section (a)
print("Section (a): Network initialization and training")
print("--------------------")
training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
nn = backprop_network.Network([784, 40, 10])
nn.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

# Section (b)
print("Section (b):")
print("Plots for training accuracy, training loss (ℓ(W)) and test accuracy across epochs")
print("--------------------")
learning_rates = [0.001, 0.01, 0.1, 1, 10, 100]
training_loss = []
test_accuracy = []
training_accuracy = []
training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
for rate in learning_rates:
    nn = backprop_network.Network([784, 40, 10])
    cur_training_accuracy, cur_training_loss, cur_test_accuracy = nn.SGD_for_question2(training_data, epochs=30,
                                                                                       mini_batch_size=10,
                                                                                       learning_rate=rate,
                                                                                       test_data=test_data)
    training_accuracy.append(cur_training_accuracy)
    training_loss.append(cur_training_loss)
    test_accuracy.append(cur_test_accuracy)

# Plotting
plot(learning_rates, training_accuracy, "Accuracy", "Training Accuracy")
plot(learning_rates, training_loss, "Loss", "Training Loss")
plot(learning_rates, test_accuracy, "Accuracy", "Test Accuracy")

# Section (c)
print("Section (c): train the network on the whole training set and test on the whole test set")
print("--------------------")
training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
nn = backprop_network.Network([784, 40, 10])
nn.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

# Section (d)
print("Section (d): Bonus question")
print("--------------------")
training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
nn = backprop_network.Network([784, 2048, 10])
nn.SGD(training_data, epochs=30, mini_batch_size=5, learning_rate=0.1, test_data=test_data)
