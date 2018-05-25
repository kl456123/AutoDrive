# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import os

filename_pairs = []

dir_path = os.path.expanduser('~/AVOD')

for file in os.listdir(dir_path):
    filename_pairs.append((os.path.join(dir_path, file), 1))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


tfrecords_filename = 'demo.tfrecords'

original_images = []


def encode():

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for img_path, label in filename_pairs:
        img = np.array(Image.open(img_path))
        height = img.shape[0]
        width = img.shape[1]

        original_images.append([img, label])
        img_raw = img.tostring()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'filename':
                    _bytes_feature(os.path.basename(img_path.encode('utf8'))),
                    'height':
                    _int64_feature(height),
                    'width':
                    _int64_feature(width),
                    'image_raw':
                    _bytes_feature(img_raw),
                    'label':
                    _int64_feature(label)
                }))
        writer.write(example.SerializeToString())

    writer.close()


recontructed_images = []


def decode():
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        filename = example.features.feature['filename'].bytes_list.value[
            0].decode('utf8')
        print(filename)
        img_string = (
            example.features.feature['image_raw'].bytes_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])
        recontructed_image = np.frombuffer(
            img_string, dtype=np.uint8).reshape((height, width, -1))
        recontructed_images.append((recontructed_image, label))


# encode()
decode()
