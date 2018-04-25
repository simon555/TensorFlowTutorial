import tensorflow as tf
import matplotlib.pyplot as pl
import numpy as np
import time

Nepoch=3
Nbatch=200
N=3000


#generate fake data
X_f = np.linspace(0,10,N).reshape((N,1))
Y_f = 3.14 * X_f + np.random.random(N).reshape((N,1))


#create a tf dataset 
#repeat allows to go through the dataset many epoch-times
dataset=tf.data.Dataset.from_tensor_slices({'x':X_f, 'y':Y_f}).repeat()
#buffer_size matters! 
dataset = dataset.shuffle(buffer_size=1000)
#specify the batch size
dataset = dataset.batch(Nbatch)

#play the role of placeholder
iterator=dataset.make_one_shot_iterator()
inputs = iterator.get_next()


#define the input type
X=inputs['x']


#define the model
class Model:
    def __init__(self, 
                 Nneurons=10):
        self.dense1=tf.layers.Dense(1)
        #self.dense2=tf.layers.Dense(1)        
        
    
    def __call__(self,x):
        output=self.dense1(x)
        #output=self.dense2(output)
        return(output)

model=Model()

#define the cost function
Y=model(X)
Y_label=inputs['y']
cost=tf.losses.mean_squared_error(Y_label,Y)


#define the optimizer
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 


#open session
sess=tf.Session()

#init the weights in the model
init= tf.global_variables_initializer()

# initialize the weigths
sess.run(init)

#keep track of the loss function
loss = []

#clean figure plots
pl.clf()


#run optimization
for epoch in range(Nepoch):
    for batchIteraton in range(N//Nbatch):
        #take what matters in the computation graph
        _,tmpLoss=sess.run([train_op,cost])
        
        #track the loss function
        loss+=[tmpLoss]
       
    #pl.gcf.clear()
    pl.plot(loss)
    pl.xlabel('number of batch feedforward')
    pl.ylabel('L2 loss')
    pl.show()
    