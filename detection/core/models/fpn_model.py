# -*- coding: utf-8 -*-

from detection.core.model import DetectionModel
from detection.core.model_utils import resnet_util
import tensorflow as tf

slim = tf.contrib.slim


class FPNModel(DetectionModel):
    KEY_FPN_P2 = 'fpn_p2'
    KEY_FPN_P3 = 'fpn_p3'
    KEY_FPN_P4 = 'fpn_p4'
    KEY_FPN_P5 = 'fpn_p5'
    KEY_FPN_P6 = 'fpn_p6'

    def __init__(self, model_config, dataset,training=True):
        super().__init__(model_config)
        self.backbone = model_config.backbone
        self.backbone in ['resnet50', 'resnet101']
        loss_config = model_config['loss']
        self._weight_decay = loss_config['weight_decay']

        self.is_training = model_config == 'train'

    def _build_resnet50(self, inputs=None):
        return resnet_util.resnet_graph(inputs, self.backbone,
                                        self.is_training, self.backbone, self._weight_decay)

    def _build_pyramid(self, Cn):
        C1, C2, C3, C4, C5 = Cn
        with tf.variable('Pyramid'):
            P5 = slim.conv2d(C5, 256, (1, 1), stride=1, scope='fpn_c5p5')
            P4 = tf.add(
                slim.conv2d_transpose(P5, (2, 2), scope='fpn_p5upsampled'),
                slim.conv2d(C4, 256, (1, 1), scope='fpn_c4p4'),
                name='fpn_p4add')

            P3 = tf.add(
                slim.conv2d_transpose(P4, (2, 2), scope='fpn_p4upsampled'),
                slim.conv2d(C3, 256, (1, 1), scope='fpn_c3p3'),
                name='fpn_p3add')
            P2 = tf.add(
                slim.conv2d_transpose(P3, (2, 2), scope='fpn_p3upsampled'),
                slim.conv2d(C2, 256, (1, 1), scope='fpn_c2p2'),
                name='fpn_p2add')

            P2 = slim.conv2d(
                P2, 256, (3, 3), padding='SAME', scope=self.KEY_FPN_P2)
            P3 = slim.conv2d(
                P3, 256, (3, 3), padding='SAME', scope=self.KEY_FPN_P3)
            P4 = slim.conv2d(
                P4, 256, (3, 3), padding='SAME', scope=self.KEY_FPN_P4)
            P5 = slim.conv2d(
                P5, 256, (3, 3), padding='SAME', scope=self.KEY_FPN_P5)

            P6 = slim.max_pool2d(P5, (1, 1), 2, scope=self.KEY_FPN_P6)

            return [P2, P3, P4, P5, P6], [P2, P3, P4, P5]

    def build(self, inputs=None):
        Cn = self._build_resnet50(inputs)
        return self._build_pyramid(Cn)
        # rpn_feature_maps, mrcnn_feature_maps = self._build_pyramid(Cn)
        # prediction_dict = dict(
        # zip(['rpn_feature_maps', 'mrcnn_feature_maps'],
        # [rpn_feature_maps, mrcnn_feature_maps]))
        # return prediction_dict

    def loss(self):
        pass
