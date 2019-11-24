# Example code wihch is uploaded at Tensorflow Official Site.
# test for GPU settings
# https://www.tensorflow.org/guide/gpu

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.debugging.set_log_device_placement(True)

# 텐서 생성
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)
