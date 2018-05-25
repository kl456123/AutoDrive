# -*- coding: utf-8 -*-

import os
import tensorflow as tf


def load_checkpoints(checkpoint_dir, saver):
    all_checkpoint_states = tf.train.get_checkpoint_state(checkpoint_dir)
    if all_checkpoint_states:
        all_checkpoint_paths = all_checkpoint_states.all_model_checkpoint_paths
        saver.recover_last_checkpoints(all_checkpoint_paths)
    else:
        print("No checkpoints found!")
