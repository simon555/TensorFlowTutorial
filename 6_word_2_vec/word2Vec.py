# -*- coding: utf-8 -*-

import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pl


# Configuration
BATCH_SIZE = 32
WINDOW_SIZE = 2
# Dimension of the embedding vector. Two too small to get
# any meaningful embeddings, but let's make it 2 for simple visualization
embedding_size = 2
num_sampled = 16 # Number of negative examples to sample.



#building the dataset
# Sample sentences
sentences = ["the quick brown fox jumped over the lazy dog",
            "I love cats and dogs",
            "we all love cats and dogs",
            "cats and dogs are great",
            "sung likes cats",
            "she loves dogs",
            "cats can be very independent",
            "cats are great companions when they want to be",
            "cats are playful",
            "cats are natural hunters",
            "It's raining cats and dogs",
            "dogs and cats love sung"]

# sentences to words and count
words = " ".join(sentences).split()
count = collections.Counter(words).most_common()
print ("Word count", count[:5])


# =============================================================================
# build one hot embedding
# =============================================================================
# Build dictionaries
rdic = [i[0] for i in count] #reverse dic, idx -> word
dic = {w: i for i, w in enumerate(rdic)} #dic, word -> id
voc_size = len(dic)


# Make indexed word data
data = [dic[word] for word in words]
print('Sample data', data[:10], [rdic[t] for t in data[:10]])


# Let's make a training data for window size 1 for simplicity
# ([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...

# Let's make a training data for window size 1 for simplicity
# ([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...
cbow_pairs = [];
for i in range(1, len(data)-1) :
    cbow_pairs.append([[data[i-1], data[i+1]], data[i]]);
print('Context pairs', cbow_pairs[:10])

# Let's make skip-gram pairs
# (quick, the), (quick, brown), (brown, quick), (brown, fox), ...
skip_gram_pairs = [];
for c in cbow_pairs:
    for index in range(WINDOW_SIZE):
        skip_gram_pairs.append([c[1], c[0][index]])
print('skip-gram pairs', skip_gram_pairs[:5])

skipArray=np.array(skip_gram_pairs)

inputWords=skipArray[:,0]
outputWords=skipArray[:,1].reshape(-1,1)

#inputWords=skipArray[:,0].reshape(-1,1)
#outputWords=skipArray[:,1].reshape(-1,1)


#create a tf dataset 
#repeat allows to go through the dataset many epoch-times
dataset=tf.data.Dataset.from_tensor_slices({'inputWord':inputWords, 'windowWord':outputWords}).repeat()
#buffer_size matters! 
dataset = dataset.shuffle(buffer_size=1000)
#specify the batch size
dataset = dataset.batch(BATCH_SIZE)
#play the role of placeholder
iterator=dataset.make_one_shot_iterator()
inputs = iterator.get_next()


#define the inputs/labels from the dataset
X=inputs['inputWord']
Y_label=inputs['windowWord']


voc_size=len(words)


# Ops and variables pinned to the CPU because of missing GPU implementation
    # Look up embeddings for inputs.
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, X) # lookup table




# Construct the variables for the NCE loss
nce_weights = tf.Variable(
    tf.random_uniform([voc_size, embedding_size],-1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))



# Compute the average NCE loss for the batch.
# This does the magic:
#   tf.nn.nce_loss(weights, biases, inputs, labels, num_sampled, num_classes ...)
# It automatically draws negative samples when we evaluate the loss.
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, Y_label, embed, num_sampled, voc_size))



# Use the adam optimizer
train_op = tf.train.AdamOptimizer(1e-1).minimize(loss)

# Launch the graph in a session
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)


# Initializing all variables
init=tf.global_variables_initializer()
sess.run(init)

for step in range(1000):
    _, loss_val = sess.run([train_op, loss])
    if step % 10 == 0:
      print("Loss at ", step, loss_val) # Report the loss

# Final embeddings are ready for you to use. Need to normalize for practical use
trained_embeddings = embeddings.eval(session=sess)

# Show word2vec if dim is 2
if trained_embeddings.shape[1] == 2:
    labels = rdic[:10] # Show top 10 words
    for i, label in enumerate(labels):
        x, y = trained_embeddings[i,:]
        pl.scatter(x, y)
        pl.annotate(label, xy=(x, y), xytext=(5, 2),
            textcoords='offset points', ha='right', va='bottom')
pl.savefig("word2vec.png")

