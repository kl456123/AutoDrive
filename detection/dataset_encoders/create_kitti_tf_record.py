# -*- coding: utf-8 -*-
r"""Convert raw KITTI detection dataset to TFRecord for object_detection.

Converts KITTI detection dataset to TFRecords with a standard format allowing
  to use this dataset to train object detectors. The raw dataset can be
  downloaded from:
  http://kitti.is.tue.mpg.de/kitti/data_object_image_2.zip.
  http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip
  Permission can be requested at the main website.

  KITTI detection dataset contains 7481 training images. Using this code with
  the default settings will set aside the first 500 images as a validation set.
  This can be altered using the flags, see details below.

Example usage:
    python object_detection/dataset_tools/create_kitti_tf_record.py \
        --data_dir=/home/user/kitti \
        --output_path=/home/user/kitti.record
"""

import hashlib
import io
import os

import numpy as np
import PIL.Image as pil
import tensorflow as tf

from detection.utils import dataset_util
from detection.utils import label_map_util
# from detection.utils.np_box_ops import iou

tf.app.flags.DEFINE_string('data_dir', '',
                           'Location of root directory for the '
                           'data. Folder structure is assumed to be:'
                           '<data_dir>/training/label_2 (annotations) and'
                           '<data_dir>/data_object_image_2/training/image_2'
                           '(images).')

tf.app.flags.DEFINE_string(
    'output_path', '', 'Path to which TFRecord files'
    'will be written. The TFRecord with the training set'
    'will be located at: <output_path>_train.tfrecord.'
    'And the TFRecord with the validation set will be'
    'located at: <output_path>_val.tfrecord')

tf.app.flags.DEFINE_string('classes_to_use', 'car,pedestrian,dontcare',
                           'Comma separated list of class names that will be'
                           'used. Adding the dontcare class will remove all'
                           'bboxs in the dontcare regions.')

tf.app.flags.DEFINE_string('label_map_path', 'data/kitti_label_map.pbtxt',
                           'Path to label map proto.')

tf.app.flags.DEFINE_string('train_set_file', 'val.txt',
                           'val set of images to'
                           'be used as a train set.')

FLAGS = tf.app.flags.FLAGS


def convert_kitti_to_tfrecords(data_dir, output_path, classes_to_use,
                               label_map_path, train_set_file):

    label_map_dict = label_map_util.load_labelmap(label_map_path)

    annotation_dir = os.path.join(data_dir, 'training', 'label_2')

    image_dir = os.path.join(data_dir, 'data_object_image_2', 'training',
                             'image_2')
    calib_dir = os.path.join(data_dir, 'data')

    velo_dir = os.path.join(data_dir, 'training', 'velodyne')

    train_writer = tf.python_io.TFRecordWriter(
        '{}_train.tfrecord'.format(output_path))
    val_writer = tf.python_io.TFRecordWriter(
        '{}_val.tfrecord'.format(output_path))

    ##TODO finish it!

    with open(train_set_file, "r") as f:
        lines = f.readlines()

    train_writer.close()
    val_writer.close()


def get_vals_and_trains(data_split_dir, train_set_file):
    pass


def read_annotation_file(filename):
    """Reads a KITTI annotation file.

  Converts a KITTI annotation file into a dictionary containing all the
  relevant information.

  Args:
    filename: the path to the annotataion text file.

  Returns:
    anno: A dictionary with the converted annotation information. See annotation
    README file for details on the different fields.
  """
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content]

    anno = {}
    anno['type'] = np.array([x[0].lower() for x in content])
    anno['truncated'] = np.array([float(x[1]) for x in content])
    anno['occluded'] = np.array([int(x[2]) for x in content])
    anno['alpha'] = np.array([float(x[3]) for x in content])

    anno['2d_bbox_left'] = np.array([float(x[4]) for x in content])
    anno['2d_bbox_top'] = np.array([float(x[5]) for x in content])
    anno['2d_bbox_right'] = np.array([float(x[6]) for x in content])
    anno['2d_bbox_bottom'] = np.array([float(x[7]) for x in content])

    anno['3d_bbox_height'] = np.array([float(x[8]) for x in content])
    anno['3d_bbox_width'] = np.array([float(x[9]) for x in content])
    anno['3d_bbox_length'] = np.array([float(x[10]) for x in content])
    anno['3d_bbox_x'] = np.array([float(x[11]) for x in content])
    anno['3d_bbox_y'] = np.array([float(x[12]) for x in content])
    anno['3d_bbox_z'] = np.array([float(x[13]) for x in content])
    anno['3d_bbox_rot_y'] = np.array([float(x[14]) for x in content])

    return anno


def main(_):
    convert_kitti_to_tfrecords(
        data_dir=FLAGS.data_dir,
        output_path=FLAGS.output_path,
        classes_to_use=FLAGS.classes_to_use.split(','),
        label_map_path=FLAGS.label_map_path,
        validation_set_file=FLAGS.validation_set_file)


if __name__ == '__main__':
    tf.app.run()
