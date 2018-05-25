# -*- coding: utf-8 -*-

from detection.core.model import DetectionModel
from detection.core.model import FPNModel
import tensorflow as tf

slim = tf.contrib.slim


class MaskRCNNModel(DetectionModel):
    ##############################
    # Key for Placeholders
    ##############################
    PL_IMG_INPUT = 'img_input_pl'
    PL_LABEL_BOXES_2D = 'label_boxes_2d_pl'
    PL_LABEL_CLASSES = 'label_classes_pl'
    PL_IMG_IDX = 'img_idx'
    PL_IMG_SHAPE = 'img_shape'
    PL_IMG_SCALE = 'img_scale'

    #############################
    # Key for Predictions
    #############################

    #############################
    # Key for Loss
    #############################

    def __init__(self, model_config, dataset):
        super().__init__(model_config)
        #############################
        # Network Input
        #############################
        # key: name ,value: placeholder
        self.placeholders = dict()

        # key: name ,value: numpy input
        self._placeholder_inputs = dict()

        self._img_depth = dataset._img_depth

        self._fpn_model = FPNModel(model_config, dataset)

    def _add_placeholders(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _build_input_pl(self):
        with tf.variable_scope('img_input'):
            img_input_placeholder = self._add_placeholders(
                tf.float32, [None, None, self._img_depth])

            self._img_input_batches = tf.expand_dims(
                img_input_placeholder, axis=0)

        with tf.variable_scope('pl_labels'):
            # classes
            self._add_placeholders(tf.float32, [None], self.PL_LABEL_CLASSES)

            # boxes_2d
            self._add_placeholders(tf.float32, [None, 4],
                                   self.PL_LABEL_BOXES_2D)

        with tf.variable_scope('img_info'):
            self._add_placeholders(tf.float32, [None], self.PL_IMG_SCALE)
            self._add_placeholders(tf.float32, [None, 2], self.PL_IMG_SHAPE)
            self._add_placeholders(tf.int32, [None], self.PL_IMG_IDX)

    def create_feed_dict(self):
        pass

    def _build_feature_map(self):
        return self._fpn_model.build(self._img_input_batches)

    def _build_(self):
        with slim.arg_scope([slim.conv2d],):


    def _generate_anchors(self):
        pass

    def build(self):
        # build input place holder
        self._build_input_pl()

        # build feature extractor
        feature_maps_dict = self._build_feature_map()
        self._

    def loss(self):
        pass
