#!/usr/bin/env python
# encoding: utf-8

from abc import ABC, abstractmethod
from tensorflow import tf


class BoxCoder(ABC):
    def __init__(self):
        pass

    def encode(self, boxes, anchors):
        """
        Args:
            boxes:
            anchors:
        Returns:

        """
        with tf.name_scope('Encode'):
            return self._encode(boxes, anchors)

    def decode(self, relative_boxes, anchors):
        """
        Args:
            relative_boxes:
            anchors:
        Returns:
        """
        with tf.name_scope('Decode'):
            return self._decode(relative_boxes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        pass

    @abstractmethod
    def _decode(self, relative_boxes, anchors):
        pass
