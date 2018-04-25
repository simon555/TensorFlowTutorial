# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pl


BATCH_SIZE=128
BUFFER_SIZE=1000
N_EPOCH=10

#load dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


#organize the data with a tf dataset

#train
trainDataset=tf.data.Dataset.from_tensor_slices({'image':trX, 'label':trY}).repeat()
trainDataset=trainDataset.shuffle(BUFFER_SIZE)
trainDataset=trainDataset.batch(BATCH_SIZE)
trainIterator=trainDataset.make_one_shot_iterator()
trainInputs = trainIterator.get_next()

#test
testDataset=tf.data.Dataset.from_tensor_slices({'image':teX, 'label':teY}).repeat()
testDataset=testDataset.shuffle(BUFFER_SIZE)
testDataset=testDataset.batch(BATCH_SIZE)
testIterator=testDataset.make_one_shot_iterator()
testIinputs = testIterator.get_next()

#checkup
#print('train dataset')
#with tf.Session() as sess: 
#    print(sess.run(trainInputs)['image'][0].shape)


#define inputs of the model
X=trainInputs['image']

#define the model
class Model:
    def __init__(self):
        self.dense1=tf.layers.Dense(32,activation=tf.nn.relu)
        self.dense2=tf.layers.Dense(64,activation=tf.nn.relu)
        self.dense3=tf.layers.Dense(10,activation=tf.nn.relu)
        
    def __cal__(self, x):
        y=self.dense1(x)
        y=self.dense2(y)
        y=self.dense3(y)
        
model=Model()

#define the output of the model
Y=model(X)

#define the loss function
Y_label=trainInputs['label']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=Y_label)) 
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(Y, 1) # at predict time, evaluate the argmax of the logistic regression
        
    

#define the session + initializer
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

#keep track of the loss
loss=[]

#run optimization
for epoch in range(N_EPOCH):
    for batch in range()






    
    







