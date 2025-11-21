# Assignment 1 - Applied Machine Learning Engineering (ECE 612)
# Name: Akash Adrashannavar
# Drexel University
# -----------------------------------------------

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

with open("appml-assignment1-dataset-v2.pkl", "rb") as f:
    data = pickle.load(f)

X = data["X"]
y = data["y"]

boundaries = np.linspace(-0.001, 0.001, 21)
fractional_change = (y - X["CAD=X_close"]) / X["CAD=X_close"]
target = np.digitize(fractional_change, boundaries)

X["weekday"] = X["date"].dt.weekday
X["hour"] = X["date"].dt.hour
X["month"] = X["date"].dt.month

ticker_features = X.drop(columns=["date", "weekday", "hour", "month"]).values.astype(np.float32)

def serialize_example(ticker_row, weekday, hour, month, label):
    feature = {
        "tickers": tf.train.Feature(float_list=tf.train.FloatList(value=ticker_row)),
        "weekday": tf.train.Feature(int64_list=tf.train.Int64List(value=[weekday])),
        "hour": tf.train.Feature(int64_list=tf.train.Int64List(value=[hour])),
        "month": tf.train.Feature(int64_list=tf.train.Int64List(value=[month])),
        "target": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

with tf.io.TFRecordWriter("dataset.tfrecords") as writer:
    for i in range(len(X)):
        example = serialize_example(
            ticker_features[i],
            int(X.iloc[i]["weekday"]),
            int(X.iloc[i]["hour"]),
            int(X.iloc[i]["month"]),
            int(target[i])
        )
        writer.write(example)
