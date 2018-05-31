#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np


from detection.core.ops import box_ops


boxes1 = np.array([[0, 0, 2, 2], [1, 1, 4, 4]])
boxes2 = np.array([[1, 1, 3, 3], [1, 1, 3, 3]])
boxes3 = np.array([[1, 1, 3, 3], [4, 4, 5, 5], [1, 1, 2, 2]])

boxes1 = tf.to_float(boxes1)
boxes2 = tf.to_float(boxes2)
boxes3 = tf.to_float(boxes3)


class ClipWindowTest(tf.test.TestCase):
    def test_clip_window(self):
        tf.placeholder()
        with self.test_session() as sess:
            sess.run()
        pass


def test_intersection():
    intersection = box_ops.intersection(boxes1, boxes2)
    return intersection


def test_union():
    union = box_ops.union(boxes1, boxes2)
    return union


def test_area():
    return box_ops.area(boxes1)


def test_iou():
    iou = box_ops.iou(boxes1, boxes2)
    return iou


def test_compute_overlaps():
    overlaps = box_ops.compute_overlaps(boxes1, boxes3)
    return overlaps


def test_clip_windows():
    anchors = np.array([[0, 0, 3, 3], [-1, -3, 4, 5], [0, 0, 5, 5]])
    anchors = tf.to_float(anchors)
    window = (4, 4)
    keep_inds = box_ops.clip_windows(anchors, window)
    return keep_inds


def main():
    intersection = test_intersection()
    union = test_union()
    iou = test_iou()
    shape = tf.shape(iou)
    overlaps = test_compute_overlaps()
    all_area = test_area()
    with tf.Session() as sess:
        inter, uni, iou_, shape_, overlaps_, all_area_ = sess.run(
            [intersection, union, iou, shape, overlaps, all_area])
    print(inter, uni, iou_)
    print("iou shape: ", shape_)
    print("overlaps", overlaps_)
    print("all area", all_area_)


main()
