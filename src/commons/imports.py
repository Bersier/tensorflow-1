import sys

import tensorflow as tf

print("TensorFlow version:", tf.version.VERSION)
print("Keras version:", tf.keras.__version__)
print("Python version: {}\n".format(sys.version))

FLOAT_TYPE = tf.float32
tf.keras.backend.set_floatx('float32')

AUTOTUNE = tf.data.experimental.AUTOTUNE
