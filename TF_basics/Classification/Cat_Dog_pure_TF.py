import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os

from random import shuffle
from tqdm import tqdm

# Convolutional Layer 1.
filter_size1 = 4 
num_filters1 = 32
# Convolutional Layer 2.
filter_size2 = 4
num_filters2 = 64
# Convolutional Layer 3.
filter_size3 = 4
num_filters3 = 128
# Convolutional Layer 4
filter_size4 = 4
num_filters4 = 256
# Convolutional Layer 5
filter_size5 = 4
num_filters5 = 128
# Fully-connected layer.
fc_size = 1024         
# Number of colo channels for the images: 1 channel for gray-scale.
num_channels = 3
# image dimensions (only squares for now)
img_size = 200
# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
# class info
classes = ['dogs', 'cats']
num_classes = len(classes)
# batch size
batch_size = 245

checkpoint_dir = "models/"

TRAIN_DIR = 'train'
TEST_DIR = 'test'

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,1)
        img = cv2.resize(img, (img_size,img_size))
#        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('dog_bgr_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]  #image id
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size,img_size))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
#test_data = process_test_data()
#train_data = np.load('dog_train_data.npy')
#test_data = np.load('test_data.npy')
#data = train_data.read_data_sets

