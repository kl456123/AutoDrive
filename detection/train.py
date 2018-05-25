# -*- coding: utf-8 -*-

import tensorflow as tf
import sys
import os

# add module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../wavedata"))

from detection.utils.config_util import get_configs_from_pipeline_file
from detection.builders.dataset_builder import DatasetBuilder

from detection.core.models.maskrcnn_model import MaskRCNNModel
from detection.core.models.fpointnet_model import FPointnetModel

from detection.core import trainer

flags = tf.app.flags

flags.DEFINE_string("pipeline_config", "", "Path to the pipeline config")

flags.DEFINE_string("device", '0', "CUDA device id")

flags.DEFINE_string("data_split", "train", "Data split for training")

FLAGS = flags.FLAGS


def train(config):

    dataset_config = config["dataset_config"]
    dataset_config.data_split = FLAGS.data_split
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config)

    model_config = config["model_config"]
    model_name = model_config.model_name
    if model_name == "fpointnet_model":
        model = FPointnetModel(model_config, dataset)
    elif model_name == "maskrcnn_model":
        model = MaskRCNNModel(model_config, dataset)

    train_config = config["train_config"]
    trainer.train(model, train_config)


def main(argv):
    assert not FLAGS.pipeline_config == "", "please determinate pipeline config first! "
    config = get_configs_from_pipeline_file(FLAGS.pipeline_config)

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device

    train(config)


if __name__ == "__main__":
    tf.app.run()
