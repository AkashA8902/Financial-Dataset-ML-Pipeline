# Assignment 1 - Applied Machine Learning Engineering (ECE 612)
# Name: Akash Adrashannavar
# Drexel University
# -----------------------------------------------

import tensorflow as tf

class ImputerLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ImputerLayer, self).__init__(**kwargs)

    def call(self, inputs):
        means = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        counts = tf.reduce_sum(tf.cast(~tf.math.is_nan(inputs), tf.float32), axis=0)
        sums = tf.reduce_sum(means, axis=0)
        means = sums / tf.maximum(counts, 1)
        filled = tf.where(tf.math.is_nan(inputs), means, inputs)
        return filled
