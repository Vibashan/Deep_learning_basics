import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
import numpy as np  
import matplotlib.pyplot as plt

dataset = pd.read_csv('stock.csv')
#print dataset.dtypes

X = dataset[['Open','High','Low']]
print X
y = dataset['Close']
print y

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_train, y, test_size=0.1, random_state=0) 

regressor = LinearRegression() 
regressor.fit(x_train, y)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print coeff_df

y_pred = regressor.predict(X_test)

from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print "ML is Lit"


