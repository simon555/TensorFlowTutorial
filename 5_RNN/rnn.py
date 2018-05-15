import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as pl


import numpy as np
from tensorflow.examples.tutorials.mnist import input_data# -*- coding: utf-8 -*-


# =============================================================================
# Define hyperparameters
# =============================================================================
input_vec_size = lstm_size = 28
time_step_size = 28

batch_size = 128
test_size = 256
BUFFER_SIZE = 1000
BATCH_SIZE = 32
N_EPOCH=10

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# =============================================================================
# Get the data
# =============================================================================


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)

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

N_BATCH_TRAIN = trY.shape[0] // BATCH_SIZE
N_BATCH_TEST = teY.shape[0] // BATCH_SIZE


# define inputs
X=trainInputs['image']

# =============================================================================
# define model
# =============================================================================

class Model:
    def __init__(self, 
                 lstm_size=lstm_size,):
        self.RNN_cell = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
        self.dense=tf.layers.Dense(10)
        self.lstm_size=lstm_size
        
    def preProcess(self,X):
        '''
        Take an image as a 28*28 array and split it into a series of vectors of shape
        28, 28, 28 ... 28 : 28 times
        The image is now considered as a time series of 28 time steps
        each vector has a shape of 28,1
        
        
        '''
        # X, input shape: (batch_size, time_step_size, input_vec_size)
        XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
        # XT shape: (time_step_size, batch_size, input_vec_size)
        
        XR = tf.reshape(XT, [-1, self.lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)
        # XR shape: (time_step_size * batch_size, input_vec_size)
        X_split = tf.split(XR, time_step_size, 0) # split them to time_step_size (28 arrays)
        # Each array shape: (batch_size, input_vec_size)
        return(X_split)
    
    def __call__(self, x):
        processed=self.preProcess(x)
        
        out, states=rnn.static_rnn(self.RNN_cell, processed, dtype=tf.float32)
        prediction=self.dense(states.h)
        
        
        return(prediction)
        
model=Model()

Y_pred=model(X)


#define loss
Y_label=trainInputs['label']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y_label)) 
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(Y_pred, 1) # at predict time, evaluate the argmax of the logistic regression
       

#open session
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

#init model
init=tf.global_variables_initializer()
sess.run(init)

#keep track of losses
loss=[]
te_Loss=[]
accuracy = []

#run training
for epoch in range (N_EPOCH):
    epochLoss=0
    for iteration in range(N_BATCH_TRAIN):
        _,tmpLoss=sess.run([train_op, cost])
        epochLoss+=tmpLoss
        print('epoch ', epoch,' ; batch ', iteration +1, ' / ', N_BATCH_TRAIN, '   error : ', tmpLoss)

    
    loss+=[epochLoss/N_BATCH_TRAIN]
    
    
    #run on test set
    testLoss, predict_te=sess.run([cost, predict_op], feed_dict={X:teX, Y_label:teY})
    te_Loss+=[testLoss]
    
    
    #measure accuracy
    tmpAccuracy=np.mean( np.argmax(teY, axis=1) == predict_te )
    accuracy+=[tmpAccuracy]
    
    print('end of epoch ', epoch, ' accuracy on test set : ', tmpAccuracy)
        



pl.plot(loss, label='train')
pl.plot(te_Loss, label= 'test')
pl.show()

pl.plot(accuracy)
pl.show()    
        
        
        
        
        
        
