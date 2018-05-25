# -*- coding: utf-8 -*-

import tensorflow as tf

slim = tf.contrib.slim

# slim.conv2d
# convolution(inputs,
# num_outputs,
# kernel_size,
# stride=1,
# padding='SAME',
# data_format=None,
# rate=1,
# activation_fn=nn.relu,
# normalizer_fn=None,
# normalizer_params=None,
# weights_initializer=initializers.xavier_initializer(),
# weights_regularizer=None,
# biases_initializer=init_ops.zeros_initializer(),
# biases_regularizer=None,
# reuse=None,
# variables_collections=None,
# outputs_collections=None,
# trainable=True,
# scope=None):


def res_arg_scope(self, is_training, weight_decay=0.0005):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weight_regularizer=slim.l2_regularizer(weight_decay),
            biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope(
            [slim.conv2d],
                padding='SAME',
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training}) as arg_sc:
            return arg_sc


def identity_block(inputs, kernel_size, filters, block):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
    input_tensor: input tensor
    kernel_size: defualt 3, the kernel size of middle conv layer at main path
    filters: list of integers, the nb_filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    use_bias: Boolean. To use or not use a bias in conv layers.
    train_bn: Boolean. Train or freeze Batch Norm layres
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    scope = block

    with tf.variable_scope(scope):
        x = slim.conv2d(
            inputs,
            nb_filter1,
            (1, 1),
            # normalizer_fn=slim.batch_norm,
            # normalizer_params={'is_training': is_training},
            scope='branch2a')

        x = slim.conv2d(
            x,
            nb_filter2,
            (kernel_size, kernel_size),
            # normalizer_fn=slim.batch_norm,
            # normalizer_params={'is_training': is_training},
            scope='branch2b')

        x = slim.conv2d(
            x,
            nb_filter3,
            (1, 1),
            activation_fn=None,
            # normalizer_fn=slim.batch_norm,
            # normalizer_params={'is_training': is_training},
            scope='branch2c')
        x = tf.add(x, inputs)
        x = tf.nn.relu('out')
        return x


def conv_block(inputs, kernel_size, filters, block, stride=2):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
    input_tensor: input tensor
    kernel_size: defualt 3, the kernel size of middle conv layer at main path
    filters: list of integers, the nb_filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    use_bias: Boolean. To use or not use a bias in conv layers.
    train_bn: Boolean. Train or freeze Batch Norm layres
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    scope = block
    with tf.variable(scope):
        x = slim.conv2d(
            inputs,
            nb_filter1,
            (1, 1),
            stride,
            # normalizer_fn=slim.batch_norm,
            # normalizer_params={'is_training': is_training},
            scope='branch2a')
        x = slim.conv2d(
            x,
            nb_filter2,
            (kernel_size, kernel_size),
            # normalizer_fn=slim.batch_norm,
            # normalizer_params={'is_training': is_training},
            scope='branch2b')
        x = slim.conv2d(
            x,
            nb_filter3,
            (1, 1),
            # normalizer_fn=slim.batch_norm,
            # normalizer_params={'is_training': is_training},
            scope='branch2c')
        short_cut = slim.conv2d(
            inputs,
            nb_filter3(1, 1),
            stride,
            activation_fn=None,
            # normalizer_fn=slim.batch_norm,
            # normalizer_params={'is_training': is_training},
            scope='branch1')
        x = tf.add(x, short_cut)
        x = tf.nn.relu(name='out')
        return x


def resnet_graph(inputs,
                 architecture,
                 is_training,
                 stage5=True,
                 weight_decay=0.0005):
    assert architecture in ['resnet50', 'resnet101']
    scope = architecture

    with slim.arg_scope(res_arg_scope(is_training, weight_decay)):
        with tf.variable_scope(scope, 'resnet', [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope(
                [slim.conv2d, slim.max_pool2d],
                    outputs_collections=end_points_collection):
                ######################################
                # Stage1
                ######################################
                with tf.variable_scope('stage1'):
                    x = slim.conv2d(
                        inputs,
                        64,
                        (7, 7),
                        stride=2,
                        # normalizer_fn=slim.batch_norm,
                        # normalizer_params={'is_training': is_training},
                        scope='conv1')
                    C1 = x = slim.max_pool2d(
                        x, (3, 3), 2, padding='SAME', scope='pool1')

                ######################################
                # Stage2
                ######################################
                with tf.variable_scope('stage2'):
                    x = conv_block(
                        x,
                        3, [64, 64, 256],
                        'a',
                        stride=1,
                        is_training=is_training)
                    x = identity_block(x, 3, [64, 64, 256], 'b')
                    C2 = x = identity_block(x, 3, [64, 64, 256], 'c')
                ######################################
                # Stage3
                #####################################
                with tf.variable_scope('stage3'):
                    x = conv_block(x, 3, [128, 128, 512], 'a')
                    x = identity_block(x, 3, [128, 128, 512], 'b')
                    x = identity_block(x, 3, [128, 128, 512], 'c')
                    C3 = x = identity_block(x, 3, [128, 128, 512], 'd')
                #####################################
                # Stage4
                #####################################
                block_count = {'resnet50': 5, 'resnet101': 22}[architecture]
                with tf.variable_scope('stage4'):
                    x = conv_block(x, 3, [256, 256, 1024], 'b')
                    for i in range(block_count):
                        x = conv_block(x, 3, [256, 256, 1024], str(98 + i))
                    C4 = x
                #####################################
                # Stage5
                #####################################
                if stage5:
                    x = conv_block(x, 3, [512, 512, 2048], 'a')
                    x = identity_block(x, 3, [512, 512, 2048], 'b')
                    C5 = x = identity_block(x, 3, [512, 512, 2048], 'c')
                else:
                    C5 = None
                return [C1, C2, C3, C4, C5]
