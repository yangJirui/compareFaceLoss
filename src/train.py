# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

import numpy as np
import tflearn
from src.read_and_pop_data import next_batch, load_mnist


WEIGHT_DIR = 'cos_weight' # weights


def cosineface_losses(embedding, labels, out_num, w_init=None, s=30., m=0.4):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value, default is 30
    :param out_num: output class num
    :param m: the margin value, default is 0.4
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    with tf.variable_scope('cosineface_loss'):
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)

        logits = tf.matmul(embedding, weights)

        normed_weights = tf.nn.l2_normalize(weights, 0, 1e-8, name='weights_norm')
        normed_embedding = tf.nn.l2_normalize(embedding, 1, 1e-8, name='embedding_norm')

        # cos_theta - m
        cos_t = tf.matmul(normed_embedding, normed_weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        cos_logits = s*tf.where(tf.equal(labels, 1), cos_t_m, cos_t)
    return cos_logits, logits


def build_net(inputs,
              labels,
              num_classes=10,
              is_training=True,
              batch_norm_decay=0.95,
              batch_norm_epsilon=1e-5,
              batch_norm_scale=True):

    batch_norm_params = {
        "is_training": is_training,
        "decay": batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    # with slim.arg_scope([slim.conv2d],
    #                     weights_regularizer=slim.l2_regularizer(0.0001),
    #                     normalizer_fn=slim.batch_norm,
    #                     normalizer_params=batch_norm_params):
    #     with slim.arg_scope([slim.batch_norm], **batch_norm_params):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(0.0001),
                        activation_fn=tflearn.prelu):
            net = slim.conv2d(inputs, 32, [5, 5], scope='stage1/conv1')
            net = slim.conv2d(net, 32, [5, 5], scope='stage1/conv2')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='stage1/pool1')

            net = slim.conv2d(net, 64, [5, 5], scope='stage2/conv1')
            net = slim.conv2d(net, 64, [5, 5], scope='stage2/conv2')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='stage2/pool2')

            net = slim.conv2d(net, 128, [5, 5], scope='stage3/conv1')
            net = slim.conv2d(net, 128, [5, 5], scope='stage3/conv2')
            net = slim.avg_pool2d(net, [2, 2], stride=2, scope='stage3/pool3')

            net = slim.flatten(net)
            embed = slim.fully_connected(net, 2, scope='embed', activation_fn=tflearn.prelu)
            cos_logits, logits = cosineface_losses(embed, labels=labels, out_num=10,
                                                   w_init=slim.xavier_initializer(uniform=False),
                                                   s=24, m=0.2)
            return cos_logits, logits, embed


def train_net(dataset, total_steps, batch_size):

    img_plc = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='img')
    lable_plc = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='label')

    cos_logits, logits, _ = build_net(inputs=img_plc,
                                      labels=lable_plc,
                                      num_classes=10,
                                      is_training=True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=lable_plc))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cos_logits, labels=lable_plc))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1),
                                      tf.argmax(lable_plc, axis=1)), tf.float32))
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9)

    global_step = slim.get_or_create_global_step()

    ##################################

    # Method 1: when test, the accuracy will be very low
    # Note: if the network include BN layer, You should use method 2 or 3 to optimize the Network
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Method 2: every will be ok
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies([tf.group(*update_ops)]):
    #     train_op = optimizer.minimize(loss, global_step=global_step)


    # Method 3: same as Method2, but with less code.
    # train_op = slim.learning.create_train_op(loss, optimizer=optimizer, global_step=global_step)

    ####################################################

    smry_op = tf.summary.merge_all()

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    saver = tf.train.Saver(max_to_keep=80)

    with tf.Session() as sess:
        sess.run(init_op)

        smry_path = '../summary'
        smry_writer = tf.summary.FileWriter(smry_path, graph=sess.graph)

        for step in range(total_steps):

            img, label = next_batch(dataset, step, batch_size=batch_size)
            _, loss_np, accuracy_np = sess.run([train_op, loss, accuracy],
                                               feed_dict={
                                                   img_plc:img,
                                                   lable_plc: label
                                               })

            if step % 10 == 0:
                smry_str = sess.run(smry_op, feed_dict={
                    img_plc: img,
                    lable_plc: label
                })
                smry_writer.add_summary(smry_str, global_step=step)
                smry_writer.flush()

            if step % 10 == 0 :
                print ("step: {} || loss: {} || accuracy: {}".format(step, loss_np, accuracy_np))

            if step % 1000 == 0 and step !=0:
                save_ckpt = "../%s/model_%d.ckpt" % (WEIGHT_DIR, step)
                saver.save(sess, save_ckpt)

                print ("%d weights had been saved" % step  + 20 * "_")

if __name__ == '__main__':

    training_data, validataion_data, test_data = load_mnist(data_path='../data')
    train_net(training_data, 200000, batch_size=256)






