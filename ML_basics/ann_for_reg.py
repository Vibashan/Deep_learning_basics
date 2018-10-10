import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split 

dataset = pd.read_csv('stock.csv',names = ['Open','High','Low','Close'])
#print dataset.dtypes

X = dataset[['Open','High','Low']]
y = dataset['Close']

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_train, y, test_size=0.1, random_state=0) 
# split into input (X) and output (Y) variables

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(1000, input_dim=21, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# fix random seed for reproducibility
seed = 7
# NORMAL
best=1000000000
numpy.random.seed(seed)
for i in range(1,300):
	# evaluate model with standardized dataset
	estimator = KerasRegressor(build_fn=larger_model, epochs=i, batch_size=53, verbose=0)

	estimator.fit(X_train, y_train)
	y_pred = estimator.predict(X_test)

	from sklearn import metrics
	if metrics.mean_squared_error(y_test, y_pred)<best:
		best=metrics.mean_squared_error(y_test, y_pred)
		print "BESt tILL nOW"
		print "epoch ", i