# -*- coding: utf-8 -*-

import tensorflow as tf
from detection.builders import optimizer_builder
import datetime
import time
import os
from detection.core import trainer_util
from detection.core import summary_util
slim = tf.contrib.slim


def get_summary_dir(log_dir):
    datatime_str = str(datetime.datetime.now())
    return os.path.join(log_dir, 'train', datatime_str)


def train(model, train_config):

    global_step = tf.Variable(0, trainable=False, name='global_step')

    ############################
    # Get Training Configuration
    ############################
    max_iterations = train_config.max_iterations
    summary_interval = train_config.summary_interval
    checkpoint_interval = train_config.checkpoint_interval
    max_checkpoints = train_config.max_checkpoints_to_keep

    # some path
    train_dir = train_config.train_dir
    checkpoint_dir = os.path.join(train_dir, train_config.checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, model._config.model_name)
    log_dir = os.path.join(train_dir, train_config.log_dir)

    prediction_dict = model.build()

    loss, total_loss = model.loss(prediction_dict)

    summary_histograms = train_config.summary_histograms
    summary_img_images = train_config.summary_img_images

    ############################
    # Build Optimizer
    ############################
    optimizer = optimizer_builder.build(train_config.optimizer, global_step)

    ############################
    # Build Train Operation
    ############################
    with tf.variable_scope('train_op'):
        # TODO change to naive code
        train_op = slim.learning.create_train_op(
            total_loss,
            training_optimizer=optimizer,
            clip_gradient_norm=train_config.clip_gradient_norm,
            global_step=global_step)

    tf.summary.scalar('Training_Loss', train_op)

    saver = tf.train.Saver(max_to_keep=max_checkpoints, pad_step_number=True)

    allow_gpu_mem_growth = train_config.allow_gpu_mem_growth
    session_config = None
    if allow_gpu_mem_growth:
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = allow_gpu_mem_growth
    sess = tf.Session(session_config)

    # summaries
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_op = summary_util.summaries_to_keep(summaries, summary_histograms,
                                                summary_img_images)

    #######################################
    # Summary Writer
    #######################################
    summary_writer = tf.summary.FileWriter(
        get_summary_dir(log_dir), sess.graph)

    # init op
    init_op = tf.global_variables_initializer()

    if not train_config.overwrite_checkpoint:
        trainer_util.load_checkpoints(checkpoint_dir, saver)
        if len(saver.last_checkpoints) > 0:
            saver.restore(sess, saver.last_checkpoints[-1])
        else:
            sess.run(init_op)
    else:
        sess.run(init_op)

    global_step_scalar = tf.train.global_step(sess, global_step)
    last_time = time.time()
    for step in range(global_step_scalar, max_iterations + 1):
        # save checkpoint
        if step % checkpoint_interval == 0:
            global_step_scalar = tf.train.global_step(sess, global_step)
            assert global_step_scalar == step, "ERROR: something wrong with global step"
            saver.save(
                sess,
                save_path=checkpoint_path,
                global_step=global_step_scalar)

        # create feed dict
        feed_dict = model.create_feed_dict()

        # Write Summary
        if step % summary_interval == 0:
            current_time = time.time()
            time_elapsed = current_time - last_time
            last_time = current_time
            train_op_loss, summary_out = sess.run(
                [train_op, summary_op], feed_dict=feed_dict)
            print('Step {}, Total Loss {:0.3f}, Time Elapsed {0:.3f}'.format(
                step, train_op_loss, time_elapsed))
        else:
            # no necessary to record
            sess.run(train_op, feed_dict=feed_dict)

    # close writer
    summary_writer.close()
