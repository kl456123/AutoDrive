#!/usr/bin/env python
# encoding: utf-8

"""Faster RCNN box coder.

Faster RCNN box coder follows the coding schema described below:
    ty = (y - ya) / ha
    tx = (x - xa) / wa
    th = log(h / ha)
    tw = log(w / wa)
    where x, y, w, h denote the box's center coordinates, width and height
    respectively. Similarly, xa, ya, wa, ha denote the anchor's center
    coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
    center, width and height respectively.

    See http://arxiv.org/abs/1506.01497 for details.
"""
import tensorflow as tf

from detection.core.box_coder import BoxCoder
from detection.core.ops import box_ops


class FasterRCNNBoxCoder(BoxCoder):
    def __init__(self, scale_factors=None):
        super().__init__()
        if scale_factors:
            assert len(scale_factors) == 4
        for scale in scale_factors:
            assert scale > 0
        self._scale_factors = scale_factors

    def _encode(self, boxes, anchors):
        """
        Args:
            boxes: [N,4] each entry is like (x1,y1,x2,y2)
        """
        w_boxes, h_boxes, x_ctr_boxes, y_ctr_boxes = box_ops.whctrs(boxes)
        w_anchors, h_anchors, x_ctr_anchors, y_ctr_anchors = box_ops.whctrs(
            anchors)
        ty = (y_ctr_boxes - y_ctr_anchors) / h_anchors
        tx = (x_ctr_boxes - x_ctr_anchors) / w_anchors
        th = tf.log(h_boxes / h_anchors)
        tw = tf.log(w_boxes / w_anchors)

        for i, x in enumerate([ty, tx, th, tw]):
            x *= self._scale_factors[i]
        return tf.transpose(tf.stack(ty, tx, th, tw))

    def _decode(self, relative_boxes, anchors):
        pass
