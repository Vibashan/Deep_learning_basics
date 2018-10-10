import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

'''
If output is continous wrt to input it is know as supervised regression(input(study_hours,sleep), output(marks))
If output is discontinous wrt to input it is know as supervised classification(input(study_hours,sleep), output(grades))
'''
#Scaleing
X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # max test score is 100

class Neural_Network(object):
	def __init__(self):
		#Define Hyperparameters(cannot be changed)
		self.inputLayerSize = 2
		self.ouptutLayerSize = 1
		self.hiddenLayerSize = 3

		#weight parameters
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)  # (3x2) weight matrix from input to hidden layer
		self.W2 = np.random.randn(self.hiddenLayerSize, self.ouptutLayerSize) # (3x1) weight matrix from hidden to output layer

	def forward(self, X):
		#propogate input through networks
		#Z2 = X*W1 			 # dot product of X (input) and first set of 3x2 weights
		#a2 = f(Z2)			 # activation function
		#Z3 = a2*W2 		 # dot product of hidden layer (z2) and second set of 3x1 weights
		#output = f(z3) 	 # final activation function
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat

	def sigmoid(self, z):
		#Apply sigmoid activation function
		return 1/(1+np.exp(-z))

NN = Neural_Network()
yHat = NN.forward(X)

print "Predicted Output: \n" + str(yHat) 
print "Actual Output: \n" + str(y) 