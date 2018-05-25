# -*- coding: utf-8 -*-

import tensorflow as tf
slim = tf.contrib.slim


def build(optimizer_config, global_step=None):
    optimizer_type = optimizer_config.WhichOneOf('optimizer')
    optimizer = None
    if optimizer_type == 'rms_prop_optimizer':
        config = optimizer_config.rms_prop_optimizer
        tf.train.RMSPropOptimizer(
            _create_learning_rate(config.learning_rate, global_step),
            decay=config.decay,
            epsilon=config.epsilon,
            momentum=config.momentum_optimizer_value)
    elif optimizer_type == 'momentum_optimizer':
        config = optimizer_config.momentum_optimizer
        tf.train.MomentumOptimizer(
            _create_learning_rate(config.learning_rate, global_step),
            momentum=config.momentum_optimizer_value)
    elif optimizer_type == 'adam_optimizer':
        config = optimizer_config.adam_optimizer
        tf.train.AdamOptimizer(
            _create_learning_rate(config.learning_rate, global_step))
    elif optimizer_type == 'gradient_descent':
        config = optimizer_config.gradient_descent
        tf.train.GradientDescentOptimizer(
            _create_learning_rate(config.learning_rate, global_step))

    if optimizer is None:
        raise ValueError(
            'Optimizer {} is not supported'.format(optimizer_type))

    if optimizer_config.use_moving_average:
        # TODO change the following code to naive code
        optimizer = tf.contrib.opt.MovingAverageOptimizer(
            optimizer, average_decay=optimizer_config.moving_average_decay)
    return optimizer


def _create_learning_rate(learning_rate_config, global_step):
    learning_rate = None
    learning_rate_type = learning_rate_config.WhichOneOf('learning_rate')
    if learning_rate_type == 'constant_learing_rate':
        config = learning_rate_config.constant_learing_rate
        learning_rate = config.learning_rate
    elif learning_rate_type == 'exponential_decay_learning_rate':
        config = learning_rate_config.exponential_decay_learning_rate
        learning_rate = tf.train.exponential_decay(
            config.initial_learning_rate, global_step, config.decay_steps,
            config.decay_factor, config.staircase)

    if learning_rate is None:
        raise ValueError(
            'Learing_rate %s is not supported'.format(learning_rate_type))
    tf.summary.scalar('Learning_Rate', learning_rate)
    return learning_rate
