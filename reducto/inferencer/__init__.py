import os
import tensorflow as tf

from reducto.inferencer.model import ObjectDetectionModel, DetectionZoo, Yolo

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable TensorFlow contrib warning
if type(tf.contrib) != type(tf):
    tf.contrib._warning = None
