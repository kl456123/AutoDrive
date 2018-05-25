from google.protobuf import text_format
from detection.protos import pipeline_pb2
import tensorflow as tf


def get_configs_from_pipeline_file(pipeline_config_path):
    pipeline_config = pipeline_pb2.NetworkPipelineConfig()
    with tf.gfile.Open(pipeline_config_path, "r") as f:
        text_format.Merge(f.read(), pipeline_config)
    return create_configs_from_pipeline_proto(pipeline_config)


def create_configs_from_pipeline_proto(pipeline_config):
    config = {}
    config["model_config"] = pipeline_config.model_config
    config["train_config"] = pipeline_config.train_config
    config["eval_config"] = pipeline_config.eval_config
    config["dataset_config"] = pipeline_config.dataset_config
    return config


def get_configs_from_multiple_files():
    pass
