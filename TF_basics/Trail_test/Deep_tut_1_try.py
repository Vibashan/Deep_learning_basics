#!/usr/bin/env python

'''
input > weight > Hidden_layer1(Activation_func) > weigth > Hidden_layer2(Activation_func) > weigths > Output

At the end of forward propg -- compare the output to intented output > cost/loss function (cross entrophy(how close are we))
 > optimizor > minimize the cost (Eg: AdamOptimizer...SGD, AdaGrad)--->(BACKPROP) ====> FEED_FROWD + BACKPROP = EPIC!!! 

'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True) 
#one_hot means in MNIST we got 10 class(0-9)... therefore one_hot says 0 = 100000000 , 1 =010000000 ...... 

#Defining no of hidden_layer and no of nodes
n_nodes_hl1 = 500
n_nodes_hl2 = 1000
n_nodes_hl3 = 500

n_classes = 10   #(0-9)
batch_size = 100 #(batches of 100 images are feed and manipulate the weights and then take next set of batch and again manipulate weight)

#x -input data of matrix of height none(1) and width is 28*28 = 784
#y is the label of the of data
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	#weighst here are tensorflow variable, that variable is tensorflow random normal we r gonna specify the shape of the normal 
	# Sum of all (Input_data * weights + biases) > fed into activation func 
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}

    #(Input_data * weights + biases)
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) #Activation func(Threshold func)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    
    #softmax_cross_entropy_with_logits is the cost function
    #Its the calculate the difference between prediction and label
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost) #has default learning rate =0. 001
    
    hm_epochs = 20
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):# total no of dataset / batch_size
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
