# -*- coding: utf-8 -*-

from detection.core.model import DetectionModel
from detection.core.model import FPNModel
from detection.core.model_utils.anchor_generator import AnchorGenerator
from detection.core.model_utils.box_encoder import BBoxDecoder
from detection.core.mdoel_utils import rpn_util
from detection.core.ops import box_ops
from detection.core.ops import np2tf_ops
from detection.core.mini_batch_samplers import BalancedPositiveNegativeSampler

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

    def __init__(self, model_config, dataset, training):
        super().__init__(model_config)
        self._training = model_config.training

        #############################
        # Network Input
        #############################
        # key: name ,value: placeholder
        self.placeholders = dict()

        # key: name ,value: numpy input
        self._placeholder_inputs = dict()

        self._img_depth = dataset._img_depth

        self._fpn_model = FPNModel(model_config, dataset, training)
        self._weight_decay = self._fpn_model._weight_decay

        rpn_config = model_config['rpn_config']

        self._rpn_positive_overlaps = rpn_config.rpn_negative_overlap
        self._rpn_negative_overlaps = rpn_config.rpn_positive_overlap
        self._rpn_clobber_positives = rpn_config.rpn_clobber_positives
        self._rpn_mini_batch_size = rpn_config.rpn_mini_batch_size

        self._mb_sampler = BalancedPositiveNegativeSampler(
            rpn_config.positive_fraction)
        # anchors parameters
        anchor_scales = rpn_config['anchor_scales']
        anchor_ratios = rpn_config['anchor_ratios']
        anchor_stride = rpn_config['anchor_stride']

        self._anchor_generator = AnchorGenerator(
            anchor_scales, anchor_ratios, anchor_stride)

        self._bbox_decoder = BBoxDecoder()

        self._rpn_num_anchors_per_location = [
            len(scale) * len(anchor_ratios) for scale in anchor_scales]

        self._rpn_output_channels = 256

        self._rpn_anchor_stride = rpn_config['anchor_stride']

        # key: image shape,value: anchors in pyramid map
        self._anchors_cache = {}

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

        if self._training:
            with tf.variable_scope('pl_labels'):
                # classes
                self._gt_labels = self._add_placeholders(
                    tf.float32, [None], self.PL_LABEL_CLASSES)

                # boxes_2d
                self._gt_boxes = self._add_placeholders(tf.float32, [None, 4],
                                                        self.PL_LABEL_BOXES_2D)

        with tf.variable_scope('img_info'):
            self._add_placeholders(tf.float32, [None], self.PL_IMG_SCALE)
            self._img_shape = self._add_placeholders(
                tf.float32, [None, 2], self.PL_IMG_SHAPE)
            self._add_placeholders(tf.int32, [None], self.PL_IMG_IDX)

    def create_feed_dict(self):
        pass

    def _build_feature_map(self):
        return self._fpn_model.build(self._img_input_batches)

    def _build_rpn(self, rpn_feature_maps):
        # with slim.arg_scope([slim.conv2d],):
            # pass
        return rpn_util.rpn_graph(rpn_feature_maps, self._rpn_num_anchors_per_location, self._weight_decay)

    def _compute_featmap_shape_list(self, rpn_feature_maps):
        featmap_shape_list = []
        for Pi in rpn_feature_maps:
            featmap_shape_list.append(tf.shape(Pi)[1:3])
        return featmap_shape_list

    def _generate_anchors(self, featmap_shape_list):
        # image_shape = tuple(image_shape)
        # just return it
        # if image_shape in self._anchors_cache:
            # return self._anchors_cache[image_shape]

        # generate anchor
        # self._anchors_cache = self._anchor_generator.generate_pyramid_anchors(
            # featmap_shape_list)
        # return self._anchors_cache[image_shape]
        return self._anchor_generator.generate_pyramid_anchors(featmap_shape_list)

    def _generate_anchor_target(self, gt_boxes, anchors, img_shape):
        pass

    def _sample_rpn_mini_batch(self, bbox_overlaps):
        """
        Args:
            bbox_overlaps: (N,M) num N of anchors overlaps with num M of gt boxes
            Note that N is num of anchors after filtered by window,not num of all_anchors
        Returns:
            mb_sampled_indicator:
            mb_pos_sampled_indicator:
        """
        # shape(N,)
        max_overlaps = tf.argmax(bbox_overlaps, axis=1)

        ######################
        # neg indicator
        ######################
        negative_indicator = tf.less(max_overlaps, self._rpn_negative_overlaps)
        not_neg_indicator = tf.logical_not(negative_indicator)

        ######################
        # pos indicator
        ######################
        # fg: for each gt ,anchor with highest overlaps
        # shape(M,)
        gt_argmax_overlaps = tf.argmax(bbox_overlaps, axis=0)
        positive_indicator = tf.sparse_to_dense(gt_argmax_overlaps, tf.shape(
            max_overlaps), tf.ones_like(gt_argmax_overlaps), 0)

        # fg : above threshold IOU
        positive_indicator = tf.logical_or(tf.greater_equal(
            max_overlaps, self._rpn_positive_overlaps), positive_indicator)

        if self._rpn_clobber_positives:
            positive_indicator = tf.logical_and(
                not_neg_indicator, positive_indicator)
        else:
            positive_indicator = tf.logical_or(
                not_neg_indicator, positive_indicator)

        # care indicator(may be used for calculating loss)
        indicator = tf.logical_or(positive_indicator, negative_indicator)

        mb_sampled_indicator, mb_pos_sampled_indicator = self._mb_sampler.subsample(
            self._rpn_mini_batch_size, indicator, positive_indicator)

        return mb_sampled_indicator, mb_pos_sampled_indicator


    def _compute_target(self):
        pass

    def _generate_anchor_target_tf(self,  anchors, gt_boxes, img_shape):
        """
        Args:
            gt_boxes:
            anchors:
            img_shape:

        Return:
            rpn_bbox_inside_weights:
            rpn_bbox_outside_weights:
            rpn_bbox_labels:
            rpn_bbox_targets:
        """
        pass
        # filter anchors that inside image
        keep_inds = box_ops.clip_windows(anchors, img_shape)
        anchors_filter = tf.gather(anchors, keep_inds, axis=0)

        # match gt_boxes and anchors
        overlaps = box_ops.compute_bbox_overlaps(anchors_filter, gt_boxes)
        mb_sampled_indicator, mb_pos_sampled_indicator = self._sample_rpn_mini_batch(
            overlaps)

        #############################
        # generate anchors labels
        #############################
        # note that -1,0,1 refers to "dont care",bg,fg
        pos_labels = tf.ones_like(mb_sampled_indicator)
        negs_labels = tf.zeros_like(mb_sampled_indicator)
        dontcare = tf.fill(tf.shape(mb_sampled_indicator), -1)
        labels = tf.where(mb_sampled_indicator, negs_labels, dontcare)
        labels = tf.where(mb_pos_sampled_indicator, pos_labels, labels)

        ############################
        # generate anchors target
        ############################
        self._compute_target(anchors_filter, gt_boxes)

        # # (B,)
        # argmax_overlaps = tf.argmax(overlaps, axis=1)
        # #(G,)
        # gt_argmax_overlaps = tf.argmax(overlaps,axis=0)
        # max_overlaps = tf.reduce_max(overlaps, axis=1)
        # labels = tf.fill(tf.shape(keep_inds)[0], -1)

        # if not self._rpn_clobber_positives:
        # # like labels[max<neg_overlaps] = 0 in numpy
        # labels = np2tf_ops.bool_index(
        # max_overlaps < self._rpn_negative_overlaps, 0, labels)
        # # labels = tf.assign(tf.where(,
        # # tf.fill(tf.shape(labels), -1), labels), labels)
        # # fg
        # labels = np2tf_ops.bool_index(max_overlaps > self._rpn_positive_overlaps,1,labels)

        # if self._rpn_clobber_positives:
        # labels = np2tf_ops.bool_index(
        # max_overlaps < self._rpn_negative_overlaps, 0, labels)

    def _generate_proposal_target(self, gt_boxes, gt_labels, proposals, img_shape):
        """
        Args:
            gt_boxes:
            proposals:
            img_shape:
        Return:
            bbox_labels:
            bbox_targets:
            bbox_inside_weights:
            bbox_outside_weights:

        """
        pass

    def _generate_proposal(self, rpn_cls_pred, rpn_bbox_delta, anchors, img_shape):
        """
        Args:
            rpn_cls_pred: used for sampling bbox by scores
            rpn_bbox_delta: used for generating proposals by applying to anchors
            anchors: see before
            img_shape:
        Return:
            rois: (1*H*W*A,5) each is 5-tuple (batch_indx,x1,y1,x2,y2)
            scores: (1*H*W*A,1) classification score
        """

    def build(self):
        # build input place holder
        self._build_input_pl()

        # build feature extractor
        rpn_feature_maps, mrcnn_feature_maps = self._build_feature_map()

        # get anchors
        featmap_shape_list = self._compute_featmap_shape_list(rpn_feature_maps)
        anchors = self._generate_anchors(featmap_shape_list)

        # rpn_logits is used to get loss of it
        # rpn_cls_scores (N,all_num_anchors,2)
        rpn_bboxes_delta, rpn_cls_scores = self._build_rpn(
            rpn_feature_maps, weight_decay=self._weight_decay)

        rpn_cls_scores_reshape = tf.reshape(
            rpn_cls_scores, (tf.shape(rpn_cls_scores)[0], -1, 2))
        rpn_cls_prob = tf.softmax(rpn_cls_scores_reshape, axis=1)

        # shape(2*num_anchors_per_location,all_num_anchors)
        rpn_cls_prob_reshape = tf.reshape(
            rpn_cls_prob, (tf.shape(rpn_cls_prob)[0], -1, 2 * self._rpn_num_anchors_per_location))

        #################################
        # AnchorTarget
        # generate labels for anchors
        #################################
        if self._training:
            rpn_bbox_targets, rpn_bbox_labels, rpn_bbox_inside_weights, rpn_bbox_outside_weights = self._generate_anchor_target(
                anchors, self._gt_boxes,  self._img_shape)

        #################################
        # Proposal
        #################################
        rois = self._generate_proposal(rpn_cls_prob_reshape,
                                       rpn_bboxes_delta, anchors)
        # # decode bbox by applying deltas to anchors
        # self._bbox_decoder.decode(anchors, rpn_bboxes_delta)

        #################################
        # ProposalTarget
        #################################
        if self._training:
            self._generate_proposal_target(
                self._gt_boxes, self._gt_labels, rois,)

    def loss(self):
        pass
