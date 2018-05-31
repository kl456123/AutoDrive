#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf


from detection.core.minibatch_sampler import MinibatchSampler


class BalancedPositiveNegativeSampler(MinibatchSampler):
    def __init__(self, positive_fraction=0.5):
        if positive_fraction > 1 or positive_fraction < 0:
            raise ValueError('positive_fraction should be in range [0,1]')
        self._positive_fraction = positive_fraction

    def subsample(self, batch_size, indicator, positive_indicator):
        """
        Args:
            indicator: 1-D boolean tensor indicate those that can be subsampled
            positive_indicator: like indicator, indicate positive samples
            batch_size: num of samples in a minibatch
        Returns:
            mb_sampled: a boolean mask where True indicates samples that is in the minibatch
            mb_pos_sampled: a boolean mask where True indicates positive samples that is in the minibatch
        """

        # all pos and neg should be in the indicator
        positive_mask = tf.logical_and(positive_indicator, indicator)
        negative_mask = tf.logical_not(positive_indicator)
        negative_mask = tf.logical_and(negative_mask, indicator)

        max_num_pos = batch_size * self._positive_fraction
        # subsample from positive_mask
        sampled_pos_indicator = self.subsample_indicator(
            positive_mask, max_num_pos)

        # if num of sampled_pos is not enough, fill neg in it
        max_num_neg = batch_size - tf.reduce_sum(sampled_pos_indicator)
        sampled_neg_indicator = self.subsample_indicator(
            negative_mask, max_num_neg)

        # pos and neg
        sampled_indicator = tf.logical_or(
            sampled_neg_indicator, sampled_pos_indicator)

        return sampled_indicator, sampled_pos_indicator
