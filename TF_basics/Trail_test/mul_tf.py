#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1,x2)

#print (result)

with tf.Session() as sess:
	print(sess.run(result))