import csv
from collections import defaultdict
import numpy as np

columns = defaultdict(list) # each value in each column is appended to a list

with open('EW-MAX.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
        	if k == 'Date':
        		columns[k].append(v)# append the value into the appropriate list
        	else:
				columns[k].append(float(v))

x = np.array(columns['Date'])
print x
#print(columns['Date'])
#print(columns['Open'])
#print(columns['Close'])