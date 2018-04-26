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
trainDataset=tf.data.Dataset.from_tensor_slices({'image':trX}).repeat()
trainDataset=trainDataset.shuffle(BUFFER_SIZE)
trainDataset=trainDataset.batch(BATCH_SIZE)
trainIterator=trainDataset.make_one_shot_iterator()
trainInputs = trainIterator.get_next()

#test
testDataset=tf.data.Dataset.from_tensor_slices({'image':teX}).repeat()
testDataset=testDataset.shuffle(BUFFER_SIZE)
testDataset=testDataset.batch(BATCH_SIZE)
testIterator=testDataset.make_one_shot_iterator()
testIinputs = testIterator.get_next()




#visualizer
import matplotlib.gridspec as gridspec
def vis(images, save_name):
    dim = images.shape[0]
    n_image_rows = int(np.ceil(np.sqrt(dim)))
    n_image_cols = int(np.ceil(dim * 1.0/n_image_rows))
    gs = gridspec.GridSpec(n_image_rows,n_image_cols,top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
    for g,count in zip(gs,range(int(dim))):
        ax = pl.subplot(g)
        ax.imshow(images[count,:].reshape((28,28)))
        ax.set_xticks([])
        ax.set_yticks([])
    pl.savefig(save_name + '_vis.png')
    pl.show()


#define inputs
X=trainInputs['image']
noise = tf.random_normal(shape =tf.shape(X), mean = 0.0, stddev = 1, dtype = tf.float32) 
X_corrupted=tf.add(X,noise)
    

#define model
class Model:
    def __init__(self):
        self.conv1_e=tf.layers.Conv2D(8,3,padding='same',activation=tf.nn.relu)   
        self.conv2_e=tf.layers.Conv2D(16,3,padding='same',activation=tf.nn.relu)   
        
        self.pool = tf.layers.MaxPooling2D(2,2,'same')
        
        
        self.conv1_d=tf.layers.Conv2D(32,3,padding='same',activation=tf.nn.relu)   
        self.conv2_d=tf.layers.Conv2D(16,3,padding='same',activation=tf.nn.relu)   
        
        self.finalConv = tf.layers.Conv2D(1,3,padding='same',activation=tf.nn.relu)
                
    def getNumberFeatures(self, y):
        output=1
        for dim in y.shape[1:]:
            output*=int(dim)
        return(output)
        
    def upsample(self, x):
        dim_x,dim_y=int(x.shape[1]), int(x.shape[2])
        return(tf.image.resize_nearest_neighbor(x,[2*dim_x,2*dim_y]))

        
        
        
        
    def encode(self, x):
        y=self.conv1_e(x)
        y=self.pool(y)
        y=self.conv2_e(y)
        y=self.pool(y)
    
        return(y)
        
    def decode(self, x):
        y=self.conv1_d(x)
        y=self.upsample(y)
        y=self.conv2_d(y)
        y=self.upsample(y)
           
        y=self.finalConv(y)
        return(y)
        
        
model=Model()

code=model.encode(X)  
Y=model.decode(code)  

#define loss

cost=tf.losses.mean_squared_error(X,Y) 



#define optimizer
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer

#create session and initialize stuff
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

#keep track of the loss
tr_loss=[]
local_loss=[]
te_Loss=[]



#run optimization
for epoch in range (N_EPOCH):
    for batch_iteration in range(N_BATCH_TRAIN):
        _,tmpLoss=sess.run([train_op,cost])
        local_loss+=[tmpLoss]
        
        print('epoch {} batch {}/{} error {}'.format(epoch, batch_iteration, N_BATCH_TRAIN, tmpLoss))
    
    tr_loss+=[np.mean(np.array(local_loss))]
    local_loss=[]
    
    #run on test set
    testLoss=sess.run(cost, feed_dict={X:teX})
    te_Loss+=[testLoss]
    

#plot results
pl.plot(tr_loss)
pl.ylabel('train loss')
pl.show()

pl.plot(te_Loss)
pl.ylabel('test Loss')
pl.show()



originalImages,reconstructions=sess.run([X_corrupted,Y], feed_dict={X:teX[:100]})

vis(originalImages,'original noisy')
vis(reconstructions,'reconstructions')
print('Done')