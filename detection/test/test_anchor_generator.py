#!/usr/bin/env python
# encoding: utf-8


import os
import tensorflow as tf

from detection.core.model_utils import anchor_generator
from detection.utils.config_util import get_configs_from_pipeline_file
from detection import root_dir

pipeline_config = os.path.join(root_dir(), './config/sample.config')
config = get_configs_from_pipeline_file(pipeline_config)

model_config = anchor_generator_config = config['model_config']
anchor_generator_config = model_config.rpn_config.anchor_generator_config
featmap_shape_list = [[32, 32]]

anchor_generator = anchor_generator.AnchorGenerator(anchor_generator_config)


anchors = anchor_generator.generate_pyramid_anchors(featmap_shape_list)

print(anchors)


with tf.Session() as sess:
    np_anchors = sess.run(anchors)
    print(np_anchors)
