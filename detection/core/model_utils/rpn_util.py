#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

slim = tf.contrib.slim


def rpn_graph(self, rpn_feature_maps, num_anchors_per_location, weight_decay=0.0005):
    """
    Args:
        rpn_feature_maps:tensor,(N,H,W,C), used for region proposals

    """
    rpn_probs = []
    rpn_bboxes_delta = []
    rpn_logits = []
    with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=slim.l2_regularizer(weight_decay), activation_fn=None):
        for stage_i in enumerate(rpn_feature_maps):
            # start from 2
            with tf.variable_scope('rpn' + str(stage_i + 2)):
                shared = slim.conv2d(rpn_feature_maps, 512,
                                     kernel_size=3, stride=1, scope='shared')

                x = slim.conv2d(
                    shared, 2 * num_anchors_per_location[stage_i], kernel_size=1, stride=1, scope='rpn_class_logit')
                rpn_logit = tf.reshape(x, (-1, 2))
                rpn_logits.append(rpn_logit)

                # BG/FG
                rpn_prob = slim.softmax(rpn_logit, scope='rpn_class_probs')
                rpn_probs.append(rpn_prob)

                # box delta
                x = slim.conv2d(shared, 4 * num_anchors_per_location,
                                kernel_size=1, stride=1, scope='rpn_box_pred')
                rpn_bbox_delta = tf.reshape(x, tf.shape(x)[0], -1, 4)

                rpn_bboxes_delta.append(rpn_bbox_delta)

    all_rpn_bboxes_delta = tf.concatenate(
        rpn_bboxes_delta, axis=1, name='rpn_bboxes_delta')
    all_rpn_probs = tf.concatenate(rpn_probs, axis=1, name='rpn_probs')
    all_rpn_logits = tf.concatenate(rpn_logits, axis=1, name='rpn_logits')

    # shape (N,all_num_anchors,4)
    return all_rpn_bboxes_delta, all_rpn_logits, all_rpn_probs
