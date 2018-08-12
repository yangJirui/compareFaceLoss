# -*- coding:utf-8 -*-

from __future__ import absolute_import, print_function, division

import numpy as np
import gzip
import pickle


def load_mnist(data_path):

    file = gzip.open(data_path+ '/'+ 'mnist.pkl.gz', 'rb')

    training_data, validation_data, test_data = pickle.load(file)
    file.close()
    return training_data, validation_data, test_data


def one_hot(value, length):

    one_hot_array = np.zeros(length)

    one_hot_array[value] = 1.0

    return one_hot_array

def next_batch(dataset, step, batch_size):

    imgs, labels = dataset[0], dataset[1]
    batch_st = (step * batch_size) % len(imgs)
    batch_end = batch_st + batch_size

    batch_imgs = imgs[batch_st: batch_end]
    batch_labels = labels[batch_st: batch_end]

    batch_imgs = [img.reshape((28, 28, 1)) for img in batch_imgs]
    batch_imgs = np.array(batch_imgs)

    batch_labels = [one_hot(a_label, length=10) for a_label in batch_labels]
    batch_labels = np.array(batch_labels)

    return batch_imgs, batch_labels


if __name__ == "__main__":


    next_batch('test', 0, 100)



