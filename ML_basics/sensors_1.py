import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

columns = defaultdict(list)

with open('EW-MAX.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
          if k == 'Date':
            columns[k].append(v)# append the value into the appropriate list
          else:
        columns[k].append(float(v))

x_1 = columns['Open']
x_2 = columns['High']
x_3 = columns['Low']
x_4 = columns['Close']
x_5 = columns['Adj_Close']
x_6 = columns['Volume']

plt.ion()

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 3

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o 

  def sigmoid(self, s):
    # activation function 
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

NN = Neural_Network()
for i in xrange(1000): # trains the NN 1,000 times
  print "Actual Output: \n" + str(y) 
  print "Predicted Output: \n" + str(NN.forward(X)) 
  print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))) # mean sum squared loss
  print "\n"
  #plt.scatter(i,np.mean(np.square(y - NN.forward(X))))
  #plt.pause(0.000001)
  NN.train(X, y)

#while True:
    #plt.pause(0.000001)

