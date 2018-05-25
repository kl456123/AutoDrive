# -*- coding: utf-8 -*-

import tensorflow as tf


def summaries_to_keep(summaries, histograms=True, input_image=True):
    for summary in summaries:
        name = summary.name
        if name.startwith('histograms') and not histograms:
            summaries.remove(summary)
        if name.startwith('img_') and not input_image:
            summaries.remove(summary)

    # merge summary protocol buffer
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    return summary_op
