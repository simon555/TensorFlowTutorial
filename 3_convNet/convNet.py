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

trX=trX.reshape((-1,28,28,1))
teX=teX.reshape((-1,28,28,1))


N_BATCH_TRAIN = trY.shape[0] // BATCH_SIZE
N_BATCH_TEST = teY.shape[0] // BATCH_SIZE

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


layer=tf.layers.Conv2D(28,3,padding='same',activation=tf.nn.relu) 

#define the model
class Model:
    def __init__(self):
        self.conv1=tf.layers.Conv2D(8,3,padding='same',activation=tf.nn.relu)   
        self.conv2=tf.layers.Conv2D(16,3,padding='same',activation=tf.nn.relu)   
        self.conv3=tf.layers.Conv2D(32,3,padding='same',activation=tf.nn.relu)
        self.conv3=tf.layers.Conv2D(64,3,padding='same',activation=tf.nn.relu)
        
        self.pool = tf.layers.MaxPooling2D(2,2,'same')

        
        self.dense=tf.layers.Dense(10) 
        
    def getNumberFeatures(self, y):
        output=1
        for dim in y.shape[1:]:
            output*=int(dim)
        return(output)
        
    def __call__(self, x):
        y=self.conv1(x)
        y=self.pool(y)
        y=self.conv2(y)
        y=self.pool(y)
        y=self.conv3(y)
        y=self.pool(y)
        
        features=self.getNumberFeatures(y)
        
        y=tf.reshape(y,[-1,features])
        
        y=self.dense(y)
        
        return(y)
        
model=Model()

#define the output of the model
Y=model(X)

#define the loss function
Y_label=trainInputs['label']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=Y_label)) 
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(Y, 1) # at predict time, evaluate the argmax of the logistic regression
        
    
test=tf.multiply(cost,cost)


#define the session + initializer
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

#keep track of the loss and accuracy
loss=[]
accuracy=[]

#run optimization
for epoch in range(N_EPOCH):
    for batch in range(N_BATCH_TRAIN):
        _,tmpLoss=sess.run([train_op,cost])
        loss+=[tmpLoss]
        print('epoch ', epoch,' ; batch ', batch +1, ' / ', N_BATCH_TRAIN, '   error : ', tmpLoss)
    
    #evaluate on test set
    testPredictions=sess.run(predict_op, feed_dict={X:teX})
    
    #measure accuracy
    tmpAccuracy=np.mean( np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX}) )
    accuracy+=[tmpAccuracy]
    
    print('end of epoch ', epoch, ' accuracy on test set : ', tmpAccuracy)
        






    
    







