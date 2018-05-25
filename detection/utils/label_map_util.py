# -*- coding: utf-8 -*-

from detection.protos import string_int_labelmap_pb2
import tensorflow as tf
from google.protobuf import text_format


def load_labelmap(path):
    labelmap_proto = string_int_labelmap_pb2.StringIntLabelMap()
    with tf.gfile.Open(path, "r") as f:
        text_format.Merge(f.read(), labelmap_proto)

    return parse_labelmap(labelmap_proto)


def parse_labelmap(labelmap_proto):
    labelmap_dict = {}
    for item in labelmap_proto.items:
        labelmap_dict[item.name] = item.id

    return labelmap_dict
