#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:28:54 2019

@author: user
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import util

if __name__ == '__main__':
#def MLPkeras(in_train,label_train,testData,testLabel,numNodes,activation):
    config = tf.ConfigProto(intra_op_parallelism_threads=0, 
                        inter_op_parallelism_threads=0, 
                        allow_soft_placement=True)
    
    dataset = 'Cat-Dog'
    
    if(dataset == 'MNIST'):
        numClass = 10
        inSize = 28*28
        train = '../data/MNIST/'
        
        fileName = []
        for i in range(numClass):
            folderName = train
            folderName = folderName + '{}/'.format(i)
            fileName.append(os.listdir(folderName))
            
    if(dataset == 'Cat-Dog'):
        numClass = 2
        inSize = 200*200
        train = '../data/Cat-Dog/'
        
        
        fileName = []
        for i in range(numClass):
            folderName = train
            if(i==0):
                folderName = folderName + 'cat/'
            else:
                folderName = folderName + 'dog/'
            fileName.append(os.listdir(folderName))
    
    numLayers = 3
    numNodes = [inSize,10,numClass]
    activation = 'sigmoid'
    session = tf.Session(config=config)
    model = Sequential()
    model.add(Dense(numNodes[1], input_dim=numNodes[0], activation=activation,kernel_initializer='random_uniform',bias_initializer='zeros'))
    for i in range(2,len(numNodes)):
        model.add(Dense(numNodes[i], activation=activation,kernel_initializer='random_uniform',bias_initializer='zeros'))
        
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
    
    
    
    
            
            
    NoImages = 2
    inputs = np.zeros((NoImages*numClass,inSize))
    labMat = np.eye(10,numClass)
    labels = np.zeros((NoImages*numClass,numClass))
    
        
    for itrNo in range(20):
        folderName = train
        
        for i in range(numClass):
            if(dataset == 'MNIST'):
                fold = folderName + '{}/'.format(i)
            else:
                if(i==0):
                    fold = folderName + 'cat/'
                else:
                    fold = folderName + 'dog/'
            imgNo = 0
            files = fileName[i]
            filenames = files[itrNo*NoImages:(itrNo+1)*NoImages]
            for imgName in filenames:
                imgName = fold+imgName
                imgNo += 1
            
                
                if(dataset == 'Cat-Dog'):
                    img = util.rgb2gray(mpimg.imread(imgName))
                else:
                    img = mpimg.imread(imgName)
                inputs[imgNo,:] = img.reshape(inSize)
                labels[imgNo,:] = labMat[i]
                #itrNo += 1
            print(inputs.shape)
        model.fit(inputs, labels, epochs=20, batch_size=1,verbose=0)
        
    itrNo = itrNo+1
    for i in range(numClass):
        if(dataset == 'MNIST'):
                fold = folderName + '{}/'.format(i)
        else:
            if(i==0):
                fold = folderName + 'cat/'
            else:
                fold = folderName + 'dog/'
        imgNo = 0
        files = fileName[i]
        filenames = files[itrNo*NoImages:(itrNo+1)*NoImages]
        for imgName in filenames:
            imgName = fold+imgName
            imgNo += 1
        
            if(dataset == 'Cat-Dog'):
                img = util.rgb2gray(mpimg.imread(imgName))
            else:
                img = mpimg.imread(imgName)
            inputs[imgNo,:] = img.reshape(inSize)
            labels[imgNo,:] = labMat[i]
            #itrNo += 1
    
    #model.fit(in_train, label_train, epochs=20, batch_size=1,verbose=0)
    predicted = model.predict_classes(inputs)
    scores = model.evaluate(inputs,labels)
    
    accuracy = util.getAccuracy(np.argmax(labels,axis=1),predicted)
    f1_macro,f1_micro = util.fi_macro_micro(np.argmax(labels,axis=1),predicted, numClass)
    print(accuracy, f1_macro, f1_micro)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))