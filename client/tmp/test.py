from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import json
import requests
import tensorflow as tf
import numpy as np


def get_prediction(server_host='127.0.0.1', server_port=9000):
    loaded = tf.saved_model.load("gs://kube-1122/t-churn/model/export/1")
    print(loaded)
    print(loaded)


if __name__ == '__main__':
    get_prediction()

