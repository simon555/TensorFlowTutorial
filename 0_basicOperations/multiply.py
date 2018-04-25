# -*- coding: utf-8 -*-

import tensorflow as tf


# define your input nodes of your graph
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

#define your operation
z=tf.multiply(x,y)

#open your session
sess = tf.Session()

#run the graph with the specified inputs
output=sess.run(z, feed_dict={x:[1],y:[2]})

#checkout
print(output)

