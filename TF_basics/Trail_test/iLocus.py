import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import cross_validation



def distance(a, b):
    return tf.abs(a-b)

def linear_layers(X, n_input, n_output, activation, scope):
    with tf.variable_scope(scope or 'linear'):
        W = tf.get_variable(
            name = 'W',
            shape = [n_input, n_output],
            initializer = tf.random_normal_initializer(mean = 0.0, stddev=0.1))
        b = tf.get_variable(
            name = 'b',
            shape = [n_output],
            initializer = tf.constant_initializer())
        h = tf.matmul(X,W) + b
        if activation is not None:
            h = activation(h)
        return h

df=pd.read_csv('Data base neural.txt')
data = np.array(df)
np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data);
y=np.array(data[:,[0,1,2,3,4,5,6,7,8]])
x=np.array(np.delete(data, [0,1,2,3,4,5,6,7,8,], 1))
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(x,y,test_size=0.1)




X = tf.placeholder(dtype = np.float32, shape = [None, 4], name = 'X')
Y = tf.placeholder(dtype = np.float32, shape = [None, 9], name = 'Y')

neurons = [4, 64, 64, 64,64,64,64,64,64, 9]
current_input = X

for iter_i in range(1, len(neurons)):
    current_input = linear_layers(
        X = current_input,
        n_input = neurons[iter_i-1],
        n_output = neurons[iter_i],
        activation = tf.nn.relu if (iter_i+1) < len(neurons) else None,
        scope = "Layer_k"+str(iter_i))
    y_pred = current_input

cost = tf.reduce_mean(tf.reduce_sum(distance(y_pred, Y), 1))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

n_iter = 100
batch_size = 60

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    prev_training_cost = 0.0
    for it_i in range(n_iter):
        idxs = np.random.permutation(range(len(X_train)))
        #print(len(idxs))
        n_batches = int(len(idxs) // batch_size)
        #print (n_batches)

        for nth_batch in range(n_batches):
            idxs_i = idxs[nth_batch * batch_size: (nth_batch + 1) * batch_size]
            sess.run(optimizer, feed_dict = {X: X_train[idxs_i], Y: Y_train[idxs_i]})

        training_cost = sess.run(cost, feed_dict = {X: X_train, Y: Y_train})
        ys_pred = y_pred.eval(feed_dict = {X:X_train}, session = sess)
        

        print(training_cost, ys_pred)
    correct=tf.equal(tf.argmax(ys_pred,1),tf.argmax(Y_train,1))
    accuracy=tf.reduce_mean(tf.cast(correct,'float'))
    print(sess.run(accuracy,feed_dict={X:X_test,Y:Y_test}))





























        
    

