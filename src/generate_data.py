#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 17:09:37 2018

@author: mostafa
"""

import numpy as np
import cv2
from random import shuffle
from copy import *

def generate_true_combinations(data):
    """
    this function generates all possible similar faces from the input data.
    input:
        - data: is a list of lists, each inner list contains all the similar images for the same person, for example [[img1_path, img2_path, ...], [], ...].
    output:
        - true_combinations: is a list of pairs of similar images, for example [(img1, img2), ...].
    """
    true_combinations = []
    for group in data:
        for i in range(len(group)):
            for j in range(i, len(group)):
                true_combinations.append((group[i], group[j], 1))
    return true_combinations

def generate_false_combination(data, n_samples_per_class, n_images_per_class):
    """
    this function generates n_samples possible of dissimilar faces from the input data.
    input:
        - data: is a list of lists, each inner list contains all the similar images for the same person, for example [[img1_path, img2_path, ...], [], ...].
        - n_samples_per_class: is the maximum number of dissimilar pairs to be generated per class. 
        - n_images_per_class: is the number of images per class.
    output:
        - false_combinations: is a list of pairs of dissimilar images, for example [(img1, img2), ...].
    """
    img_indices = range(n_images_per_class)
    data_indices = list(range(len(data)))
    false_combinations = []
    for j, group in enumerate(data):
        source_images = group.copy()
        np.random.shuffle(source_images)
        for i in range(n_samples_per_class):
            index = i % len(group)
            image1 = source_images[index]
            dest_images = data_indices[:j] + data_indices[j + 1:]
            image2_class = np.random.choice(dest_images)
            image2_index = np.random.choice(img_indices)
            false_combinations.append((image1, data[image2_class][image2_index], 0))
    return false_combinations

def read_image(path):
    return cv2.imread(path, 0)

def reshape(data):
    X1 = []
    X2 = []
    Y = []
    for i in range(len(data)):
        X1.append(data[i][0])
        X2.append(data[i][1])
        Y.append(data[i][2])
    
    X1 = np.array(X1,dtype=np.float32)
    X1 = X1.reshape(X1.shape[0], 1, X1.shape[1], X1.shape[2])
    
    X2 = np.array(X2,dtype=np.float32)
    X2 = X2.reshape(X2.shape[0], 1, X2.shape[1], X2.shape[2])
    
    Y = np.array(Y,dtype=np.float32).reshape(-1,1)
    
    return X1, X2, Y
        
def prepare_to_classifier(true_combinations_train, false_combinations_train, 
                          true_combinations_test, false_combinations_test):
    # merge
    train = true_combinations_train + false_combinations_train
    test = true_combinations_test + false_combinations_test
    
    # shuffle
    shuffle(train)
    shuffle(test)
    
    # reshape train
    X1_train, X2_train, Y_train = reshape(train)
    # reshape test
    X1_test, X2_test, Y_test = reshape(test)
    
    return X1_train, X2_train, Y_train, X1_test, X2_test, Y_test

def generate_train_test_data(data_dir = '../../att_faces'):
    """
    this function works with 'AT&T Database of Faces' structure, to generate training and testing sets, 
    that will be used to train and evaluate the machine learning model.
    input:
        - data_dir: the path of the data set.
    output:
        - true_combinations_train: is a list of pairs of similar images, 
        that will be used for training, for example [(img1, img2), ...].
        - false_combinations_train: is a list of pairs of similar images, 
        that will be used for training, for example [(img1, img2), ...].
        - true_combinations_test: is a list of pairs of similar images, 
        that will be used for testing, for example [(img1, img2), ...].
        - false_combinations_test: is a list of pairs of similar images, 
        that will be used for testing, for example [(img1, img2), ...].
    """

    train_data = [ [ read_image('%s/s%d/%d.pgm'%( data_dir, i, j)) for j in range(1,11)] for i in range(1, 36)]
    test_data = [ [ read_image('%s/s%d/%d.pgm'%( data_dir, i, j)) for j in range(1,11)] for i in range(36, 41)]
    
    true_combinations_train = generate_true_combinations(train_data)
    false_combinations_train = generate_false_combination(train_data, int(len(true_combinations_train) / len(train_data)), 10)
    
    true_combinations_test = generate_true_combinations(test_data)
    false_combinations_test = generate_false_combination(test_data, int(len(true_combinations_test) / len(test_data)), 10)
    
    return prepare_to_classifier(true_combinations_train, false_combinations_train, true_combinations_test, false_combinations_test)

if __name__ == '__main__':
    X1_train, X2_train, Y_train, X1_test, X2_test, Y_test = generate_train_test_data()
