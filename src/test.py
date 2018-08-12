# -*- coding: utf-8 -*-


from __future__ import absolute_import, print_function, division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import numpy as np
from src import train
from src.read_and_pop_data import next_batch, load_mnist


WEIGHT_DIR = 'cos_weight'  # weights


def plot_embed(labels, embeds):
    colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c', 'chartreuse', 'gray', 'peru']

    for i in range(10):
        tmp_embeds = embeds[labels==i]
        print(tmp_embeds.shape)
        tmp_embeds = tmp_embeds / np.reshape(np.linalg.norm(tmp_embeds, axis=1), (-1, 1))

        plt.scatter(tmp_embeds[:, 0], tmp_embeds[:, 1], c=colors[i], alpha=0.6, s=10)
    plt.show()


def test(dataset, test_num, batch_size):

    img_plc = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='img')
    lable_plc = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='label')

    cos_logits, logits, embedding = train.build_net(inputs=img_plc,
                                        labels=lable_plc,
                                        num_classes=10,
                                        is_training=False)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1),
                                               tf.argmax(lable_plc, axis=1)), tf.float32))

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )


    restorer = tf.train.Saver()
    total_accuracy = 0.0

    fet_list = []
    label_list = []
    with tf.Session() as sess:
        sess.run(init_op)
        ckpt_path = tf.train.latest_checkpoint('../%s' % WEIGHT_DIR)
        restorer.restore(sess, save_path=ckpt_path)
        for i in range(test_num):
            img, label = next_batch(dataset, i, batch_size=batch_size)
            accuracy_np, fet = sess.run([accuracy, embedding], feed_dict={img_plc: img,
                                                                          lable_plc: label})
            total_accuracy += accuracy_np
            fet_list.append(fet)
            label_list.append(np.argmax(label, axis=1))
            print(accuracy_np)

        print("the accuracy is :: ", total_accuracy/test_num)
    fet_arry = np.concatenate(fet_list, axis=0)
    print(fet_arry)
    label_arry = np.concatenate(label_list, axis=0)
    print (fet_arry.shape)
    print (label_arry.shape)
    np.save('/home/yjr/PycharmProjects/test_slim_batch_norm/data/embedding/em2.npy', fet_arry)
    plot_embed(label_arry, fet_arry)


if __name__ == "__main__":
    training_data, validataion_data, test_data = load_mnist(data_path='../data')

    test(training_data, 1000, batch_size=30)
    '''
    
    '''