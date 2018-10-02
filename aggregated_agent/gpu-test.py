"""
https://mjgim.icim.or.kr/2017/04/30/tensorflow.html
"""

import tensorflow as tf
import os
# os.environ["CUDA_VISIBLE_DEVICES"]= str("0,1,2,3")

hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(hello))

a = tf.constant(10, name="c")
b = tf.constant(32, name="c")
print(sess.run(a + b))

