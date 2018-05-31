#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

from detection.core.ops import np2tf_ops

arr1 = tf.reshape(tf.range(10), (2, 5))
arr2 = tf.range(2)

var1 = tf.Variable(arr1, name='var1')


def slice():
    var1[var1 > 1] = 0


def test_bool_index_self():
    cond = var1 > 1
    return np2tf_ops.bool_index(cond, var1, -1)


def test_bool_index_tensor():
    cond = arr2 > 0
    return np2tf_ops.bool_index(cond, var1, 11)


def main():
    # slice()
    # var1 = test_bool_index_self()
    var1 = test_bool_index_tensor()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        var1_ = sess.run(var1)
    print(var1_)


main()
