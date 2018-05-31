#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
# use tf inplace of np
from detection.core.ops import np2tf_ops as ops

from detection.protos import anchor_generator_pb2


class AnchorGenerator(object):
    def __init__(self, config):
        """
        Args: ratios: list of float ,h/w scales: scale of base_anchor_size
            base_anchor_sizes: list of dim of anchor(square anchor)
        """
        if not isinstance(config, anchor_generator_pb2.MultipleGridAnchorGeneratorConfig):
            raise ValueError(
                'configis not of type anchor_generator_pb2.AnchorGenerator ')
        # the same for each layer
        self._scales = np.asarray(config.scales)
        self._ratios = np.asarray(config.ratios)
        if len(config.base_anchor_sizes):
            self._base_anchor_sizes = np.asarray(config.base_anchor_size)
        else:
            min_level = config.min_level
            max_level = config.max_level
            anchor_scale = config.anchor_scale
            self._base_anchor_sizes = self._generate_base_anchor_size(
                min_level, max_level, anchor_scale)
        if len(config.anchor_stride_list):
            self._anchor_stride_list = np.asarray(config.anchor_stride_list)
        else:
            # calculate online
            self._anchor_stride_list = None

        # TODO(breakpoint) check insanity

    def _generate_base_anchor_size(self, min_level, max_level, anchor_scale):
        base_anchor_sizes = []
        for level in range(min_level, max_level + 1):
            base_anchor_sizes.append(2**level * anchor_scale)
        return base_anchor_sizes

    def generate_anchors(self, base_anchor_size, featmap_shape, anchor_stride):
        """
        Args:
            scales: list of int ,anchor size in pixels
            anchor_stride:
            featmap_shape: tuple or list,(height,width) of feature map
        """
        scales, ratios = ops.meshgrid(tf.to_float(
            self._scales), tf.to_float(self._ratios))
        scales = tf.reshape(scales, [-1])
        ratios = tf.reshape(ratios, [-1])

        # enum heights and widths
        heights = scales * tf.sqrt(ratios) * base_anchor_size
        widths = scales / tf.sqrt(ratios) * base_anchor_size

        # enum position(shifts_x,shifts_y)
        shifts_x = tf.range(
            0, featmap_shape[1]) * anchor_stride
        shifts_y = tf.range(
            0, featmap_shape[0]) * anchor_stride
        shifts_x, shifts_y = ops.meshgrid(shifts_x, shifts_y)

        # dim(width)=dim(box_centers_x)=(xn,yn,len(widths))
        box_widths, box_centers_x = ops.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = ops.meshgrid(heights, shifts_y)

        # (y,x) pair and (h,w) pair
        box_centers = tf.reshape(tf.stack(
            [box_centers_y, box_centers_x], axis=2), [-1, 2])
        box_sizes = tf.reshape(tf.stack([box_heights, box_widths],
                                        axis=2), [-1, 2])

        box_centers = tf.to_float(box_centers)
        boxes = tf.concat([box_centers - 0.5 *
                           box_sizes, box_centers + 0.5 * box_sizes], axis=1)

        return boxes

    def _generate_anchor_stride_list(self, featmap_shape_list, img_shape):
        if img_shape is None:
            raise ValueError("image shape shoule not be None here! ")
        anchor_stride_list = []
        num_stages = len(self._base_anchor_sizes)
        for i in range(num_stages):
            anchor_stride_list.append(
                (img_shape[0] / featmap_shape_list[0], img_shape[1] / featmap_shape_list[1]))

        return anchor_stride_list

    def generate_pyramid_anchors(self, featmap_shape_list, img_shape=None):
        """
        Args:
            featmap_shape_list: list of tuple ,like [(height,width),(...)]
            anchor_stride_list: list of int , like [32, 64,...]
            scales: list, like [1,2,3]
        """
        assert len(featmap_shape_list) == len(
            self._base_anchor_sizes), "length of base anchors should be the same with that of featmap_shape_list"

        if self._anchor_stride_list:
            anchor_stride_list = self._anchor_stride_list
        else:
            # calculate online
            anchor_stride_list = self._generate_anchor_stride_list(
                featmap_shape_list, img_shape)
        anchors = []
        stages = len(featmap_shape_list)
        for i in range(stages):
            anchors.append(self.generate_anchors(
                self._base_anchor_sizes[i], featmap_shape_list[i], anchor_stride_list[i]))

        return anchors
