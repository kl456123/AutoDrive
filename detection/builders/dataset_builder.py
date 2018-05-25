# -*- coding: utf-8 -*-

from detection.datasets.kitti.kitti_dataset import KittiDataset


class DatasetBuilder(object):
    # static class
    CONFIG_DEFAULTS_PROTO = \
        """
        name: "kitti"
        """

    @staticmethod
    def build_kitti_dataset(dataset_config):
        return KittiDataset(dataset_config)
