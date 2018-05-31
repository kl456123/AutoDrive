#!/usr/bin/env python
# encoding: utf-8


from abc import ABC, abstractmethod

import tensorflow as tf


class MinibatchSampler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def subsample(self, indicator, batch_size, **kwargs):
        """
        subsample neg and pos to generate minibatch
        return boolean mask
        """

    @staticmethod
    def subsample_indicator(indicator, num_samples):
        """
        Args:
            indicator: 1-D tensor of boolean type
            num_samples: num of samples that should subsample from indicator
        Returns:
            selected_indicator: the same type as indicator, it indicates selected samples
        """
        indices = tf.where(indicator)
        indices = tf.random_shuffle(indices)
        indices = tf.reshape(indices, [-1])

        num_samples = tf.minimum(num_samples, tf.size(indices))
        selected_indices = indices[:num_samples]
        selected_indicator = tf.sparse_to_dense(
            selected_indices, tf.shape(indicator), tf.ones_like(selected_indices), 0)
        return selected_indicator
