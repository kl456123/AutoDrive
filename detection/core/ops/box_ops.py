#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf


def intersection(boxes1, boxes2, scope=None):
    with tf.name_scope(scope, 'Intersection', [boxes1, boxes2]):
        max_height = tf.minimum(boxes1[:, 3], boxes2[:, 3])
        min_height = tf.maximum(boxes1[:, 1], boxes2[:, 1])
        height = tf.maximum(0.0, max_height - min_height)
        max_width = tf.minimum(boxes1[:, 2], boxes2[:, 2])
        min_width = tf.maximum(boxes1[:, 0], boxes2[:, 0])
        width = tf.maximum(0.0, max_width - min_width)
        return width * height


def union(boxes1, boxes2, intersections=None, scope=None):
    with tf.name_scope(scope, 'Union', [boxes1, boxes2]):
        area1 = area(boxes1)
        area2 = area(boxes2)
        if intersections is None:
            intersections = intersection(boxes1, boxes2)
        union = (area1 + area2 - intersections)
        return union


def area(boxes, scope=None):
    """
    Args:
        boxes: (N,4) ,each element is (x1,y1,x2,y2)
    Return:
        a tensor with shape [N] representing box areas.
    """
    with tf.name_scope(scope, 'Area', [boxes]):
        return (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


def iou(boxes1, boxes2, scope=None):
    with tf.name_scope(scope, 'Iou', [boxes1, boxes2]):
        intersections = intersection(boxes1, boxes2)
        unions = union(boxes1, boxes2, intersections)

        return intersections / unions


def compute_overlaps(bboxes, gt_boxes, scope=None):
    """
    Args:
        bboxes:
        gt_boxes:
    Return:
        overlaps:
    """
    with tf.name_scope(scope, 'Overlaps', [bboxes, gt_boxes]):
        G = tf.shape(gt_boxes)[0]
        B = tf.shape(bboxes)[0]
        bboxes_broadcast = tf.reshape(tf.tile(bboxes, [1, G]), (G * B, -1))
        gt_boxes_broadcast = tf.reshape(tf.tile(gt_boxes, [B, 1]), (G * B, -1))
        overlaps = iou(bboxes_broadcast, gt_boxes_broadcast)
        overlaps = tf.reshape(overlaps, (B, G))
        return overlaps


def clip_windows(anchors, window):
    """
    Args:
        anchors:
        window:
    Return:
        keep_inds:
    """
    keep_inds = tf.where((anchors[:, 0] >= 0) &
                         (anchors[:, 1] >= 0) &
                         (anchors[:, 2] < window[0]) &
                         (anchors[:, 3] < window[1]))[:, 0]
    return keep_inds


def compute_targets(self):
    pass


def whctrs(boxes):
    """
    Args:
        boxes:
    Returns:
        width,height,center_x,center_y
    """
    width = boxes[:, 2] - boxes[:, 0] + 1.0
    height = boxes[:, 3] - boxes[:, 1] + 1.0
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    return width, height, center_x, center_y


def mkboxes(ws, hs, x_ctr, y_ctr):
    """
    Args:
        ws,hs,x_ctr,y_ctr
    Returns:
        boxes shape is (N,4)
    """
    x1 = x_ctr - 0.5 * (ws - 1.0)
    x2 = x_ctr + 0.5 * (ws - 1.0)
    y1 = y_ctr - 0.5 * (hs - 1.0)
    y2 = y_ctr + 0.5 * (hs - 1.0)
    return tf.transpose(tf.stack([x1, y1, x2, y2]))
