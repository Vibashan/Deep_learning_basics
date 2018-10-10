import numpy as np
import os

from random import shuffle
from tqdm import tqdm

TRAIN_DIR = 'flowers'
TEST_DIR = 'test'

def label_folder(folders):
	if folders == 'tulip': return [1,0,0,0,0]
	elif folders == 'rose': return [0,1,0,0,0]
	elif folders == 'dandelion': return [0,0,1,0,0]
	elif folders == 'daisy': return [0,0,0,1,0]
	elif folders == 'sunflower': return [0,0,0,0,1]

def create_train_data():
    training_data = []
    dirs = os.listdir( TRAIN_DIR )
    for folders in dirs:
    	#print(folders)
    	label = label_folder(folders)
    	req_train_dir = os.path.join(TRAIN_DIR,folders)
    	for img in tqdm(os.listdir(TRAIN_DIR)):
    	print(req_train_dir)    

train_data = create_train_data()