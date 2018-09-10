import unittest

import pygal
from tensorflow.examples.tutorials.mnist import input_data

from mnist_fun import get_accuracy
import matplotlib.pyplot as plt


class Test(unittest.TestCase):
    def test_learning_rate(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        i = 0.001
        rate = []
        accuracy = []
        while i <= 0.04:
            rate.append(i)
            accuracy.append(get_accuracy(mnist=mnist, learning_rate=i))
            i += 0.005
        # plt.bar(rate, accuracy)
        # # plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
        # plt.show()

        bar = pygal.Bar()
        bar.x_labels = map(str, rate)
        bar.add("accuracy", accuracy)
        # bar.render_in_browser()
        bar.render_to_file('learning_rate_and_accuracy.svg')

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
