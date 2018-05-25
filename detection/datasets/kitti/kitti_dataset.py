# -*- coding: utf-8 -*-

from detection.utils import label_map_util

from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.core import calib_utils

from detection.datasets.kitti import kitti_aug
from detection.core import constants
import os
import numpy as np
import itertools
import cv2


class Sample(object):
    def __init__(self, name, augs):
        self.name = name
        self.augs = augs


class KittiDataset(object):
    def __init__(self, dataset_config):
        self._cam_idx = 2
        self.config = dataset_config
        self.batch_size = dataset_config.batch_size
        self.name = dataset_config.name
        self.has_labels = dataset_config.has_labels
        self._shuffle = dataset_config.shuffle
        self._img_depth = dataset_config.img_depth

        # dataset root dir
        self.dataset_dir = dataset_config.dataset_dir

        # self._check_dataset_dir()

        # data for split
        self.data_split_dir = os.path.join(self.dataset_dir,
                                           dataset_config.data_split_dir)

        # split text
        self.data_split = os.path.join(self.dataset_dir,
                                       dataset_config.data_split + ".txt")

        # set up directories and then check
        self._set_up_directories()

        self._check_dirs_and_files()

        # dataset augmentation
        self.aug_list = dataset_config.aug_list

        # velodyne area
        self.area_extent = dataset_config.area_extent

        self.label_map = label_map_util.load_labelmap(
            dataset_config.label_map_path)

        self.num_classes = len(self.label_map.items())

        # label
        sample_names = self.load_sample_names()
        aug_sample_list = []

        for aug_num in range(len(self.aug_list) + 1):
            augmentations = list(
                itertools.combinations(self.aug_list, aug_num))
            for augmentation in augmentations:
                for sample_name in sample_names:
                    aug_sample_list.append(Sample(sample_name, augmentation))

        self.num_samples = len(aug_sample_list)
        self.sample_list = np.asarray(aug_sample_list)

        self._idx_in_epoch = 0
        self.epoch_completed = 0

    def _set_up_directories(self):
        self.image_dir = os.path.join(self.data_split_dir,
                                      "image_" + str(self._cam_idx))
        self.velo_dir = os.path.join(self.data_split_dir, "velodyne")
        self.calib_dir = os.path.join(self.data_split_dir, "calib")
        self.label_dir = os.path.join(self.dataset_dir,
                                      "training/label_" + str(self._cam_idx))

    def _check_dirs_and_files(self):
        all_files = [
            self.dataset_dir, self.data_split, self.data_split_dir,
            self.image_dir, self.velo_dir, self.calib_dir
        ]
        if self.has_labels:
            all_files.append(self.label_dir)
        for file in all_files:
            if not os.path.exists(file):
                raise FileNotFoundError(
                    "file path does not exist: {}".format(file))

    def load_sample_names(self):
        with open(self.data_split, "r") as f:
            sample_names = f.readlines()
        return np.array(sample_names)

    def parse_obj_labels(self, obj_labels, labelmap):
        label_classes = []
        label_boxes_3d = []
        label_boxes_2d = []
        for obj_label in obj_labels:
            label_boxes_3d.append([
                obj_label.t[0], obj_label.t[1], obj_label.t[2], obj_label.l,
                obj_label.w, obj_label.h, obj_label.ry
            ])
            label_boxes_2d.append(
                [obj_label.x1, obj_label.y1, obj_label.x2, obj_label.y2])
            label_classes.append(self.label_map[obj_label.type])

        return np.asarray(label_classes), np.asarray(
            label_boxes_3d), np.asarray(label_boxes_2d)

    def get_rbg_image_path(self, sample_idx):
        return os.path.join(self.image_dir, '{:06d}.png'.format(sample_idx))

    def load_samples_from_tfrecord(self, indices):
        pass

    def load_samples(self, indices):
        sample_dicts = []
        for sample_idx in indices:
            sample = self.sample_list[sample_idx]
            sample_name = sample.name

            if self.has_labels:
                obj_labels = obj_utils.read_labels(self.label_dir,
                                                   int(sample_name))

                label_classes, label_boxes_3d, label_boxes_2d = self.parse_obj_labels(
                    obj_labels, self.label_map)
            else:
                obj_labels = None
                label_classes = np.zeros(1)
                label_boxes_2d = np.zeros((1, 4))
                label_boxes_3d = np.zeros((1, 7))

            # image
            cv_bgr_image = cv2.imread(
                self.get_rbg_image_path(int(sample_name)))
            rgb_image = cv_bgr_image[..., ::-1]
            im_shape = rgb_image.shape[0:2]
            image_input = rgb_image

            # calibration
            stereo_calib_p2 = calib_utils.read_calibration(
                self.calib_dir, int(sample_name)).p2

            # point cloud
            # just project point to camera frame and then keep point in front of image
            point_cloud = obj_utils.get_lidar_point_cloud(
                int(sample_name),
                self.calib_dir,
                self.velo_dir,
                im_size=im_shape)

            #################################
            # Data Augmentation
            #################################
            if kitti_aug.AUG_FLIPPING in sample.augs:
                pass

            if kitti_aug.AUG_PCA_JITTER in sample.augs:
                pass

            sample_dict = {
                constants.KEY_IMAGE_INPUT: image_input,
                constants.KEY_POINT_CLOUD: point_cloud,
                constants.KEY_LABEL_CLASSES: label_classes,
                constants.KEY_LABEL_BOXES_2D: label_boxes_2d,
                constants.KEY_LABEL_BOXES_3D: label_boxes_3d,
                constants.KEY_STEREO_CALIB_P2: stereo_calib_p2
            }

            sample_dicts.append(sample_dict)
        return sample_dicts

    def _shuffle_samples(self):
        perm = np.arange(self.num_samples)
        np.random.shuffle(perm)
        self.sample_list = self.sample_list[perm]

    def next_batch(self):
        start = self._idx_in_epoch
        sample_in_batch = []
        if start == 0 and not self.epoch_completed and self._shuffle:
            self._shuffle_samples()

        if start + self.batch_size > self.num_samples:
            sample_in_batch.extend(
                self.load_samples(np.arange(start, self.num_samples)))
            num_rest_samples = self.num_samples - start
            if self._shuffle:
                self._shuffle_samples()
            self._idx_in_epoch = self.batch_size - num_rest_samples
            self.epoch_completed += 1
            # load the rest
            sample_in_batch.extend(
                self.load_samples(np.arange(0, self._idx_in_epoch)))
        else:
            sample_in_batch.extend(
                self.load_samples(np.arange(start, start + self.batch_size)))
            self._idx_in_epoch += self.batch_size

        return sample_in_batch
