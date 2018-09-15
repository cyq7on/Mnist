import unittest

import pygal
from tensorflow.examples.tutorials.mnist import input_data

from mnist_fun import get_accuracy
import matplotlib.pyplot as plt


class Test(unittest.TestCase):
    def test_learning_rate(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        i = 1
        rate = []
        accuracy = []
        while i <= 40:
            learning_rate = i / 1000
            rate.append(learning_rate)
            accuracy.append(get_accuracy(mnist=mnist, learning_rate=learning_rate))
            i += 5
        # plt.bar(rate, accuracy)
        # # plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
        # plt.show()

        bar = pygal.Bar()
        bar.x_labels = map(str, rate)
        bar.add("accuracy", accuracy)
        # bar.render_in_browser()
        # bar.render_to_file('learning_rate_and_accuracy.svg')
        bar.render_to_file('learning_rate_and_accuracy_for_rand.svg')
        print(rate)

    def test_plot(self):
        rate = [0.005, 0.01, 0.015, 0.02]
        accuracy = [0.91, 0.92, 0.87, 0.55]
        # plt.bar(rate, accuracy)
        # # plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
        # plt.show()

        bar = pygal.Bar()
        bar.x_labels = map(str, rate)
        bar.add("accuracy", accuracy)
        bar.render_in_browser()
        # bar.render_to_file('die_visual.svg')
