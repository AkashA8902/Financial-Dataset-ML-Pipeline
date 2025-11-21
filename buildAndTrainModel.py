# Assignment 1 - Applied Machine Learning Engineering (ECE 612)
# Name: Akash Adrashannavar
# Drexel University
# -----------------------------------------------

import tensorflow as tf
from customImputerLayerDefinition import ImputerLayer

def _parse_example(example_proto):
    feature_description = {
        "tickers": tf.io.FixedLenFeature([26], tf.float32),
        "weekday": tf.io.FixedLenFeature([], tf.int64),
        "hour": tf.io.FixedLenFeature([], tf.int64),
        "month": tf.io.FixedLenFeature([], tf.int64),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    features = {
        "tickers": parsed["tickers"],
        "weekday": parsed["weekday"],
        "hour": parsed["hour"],
        "month": parsed["month"]
    }
    return features, parsed["target"]

raw_dataset = tf.data.TFRecordDataset("dataset.tfrecords")
parsed_dataset = raw_dataset.map(_parse_example)
dataset = parsed_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

def preprocess(features):
    tickers = features["tickers"]
    weekday = tf.expand_dims(tf.cast(features["weekday"], tf.float32), axis=-1)
    hour = tf.expand_dims(tf.cast(features["hour"], tf.float32), axis=-1)
    month = tf.expand_dims(tf.cast(features["month"], tf.float32), axis=-1)
    return tf.concat([tickers, weekday, hour, month], axis=-1)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(29,)),
    ImputerLayer(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(22, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

dataset = dataset.map(lambda x, y: (preprocess(x), y))
model.fit(dataset, epochs=5)

model.save("mySavedModel")
