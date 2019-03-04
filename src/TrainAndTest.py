#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 00:45:16 2019

@author: user
"""

import MLP as mlp
import os
import numpy as np
import util as util
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

def trainAndtest(args):
    print('Training and testing in progress...')
    
    if(args.dataset != 'MNIST' and args.dataset != 'Cat-Dog'):
        print('Please provide valid dataset (MNIST / Cat-Dog)')
        exit(0)
    
    if(args.configuration == None):
        print('Please provide configuration of the network')
    
    s = args.configuration[1:-1]
    s1 = s.split(' ')
    activation = 'sigmoid'
    numLayers = len(s1)#3
    numNodes = np.zeros(numLayers, dtype=int)
    for i in range(numLayers):
        numNodes[i] = int(s1[i])
    
    if(args.dataset == 'MNIST'):
        numClass = 10
        inSize = 28*28
        
        NoImages = 0
        for i in range(10):
            folderName = args.train + '/'
            folderName = folderName + '{}/'.format(i)
            NoImages += len(os.listdir(folderName)) 
            
        #maxImage = 30
        #NoImages = maxImage*10
        inputs = np.zeros((NoImages,inSize))
        labMat = np.eye(numClass,numClass)
        labels = np.zeros((NoImages,numClass))
        
        itrNo = 0
        for i in range(numClass):
            folderName = args.train + '/'
            
            
            
            imgNo = 0
            folderName = folderName + '{}/'.format(i)
            for imgName in os.listdir(folderName):
                imgName = folderName+imgName
                imgNo += 1
                
                #if(imgNo<=maxImage):
                    
                    
                img = mpimg.imread(imgName)
                inputs[itrNo,:] = img.reshape(28*28)
                labels[itrNo,:] = labMat[i]
                itrNo += 1
    
        
        
    
    if(args.dataset == 'Cat-Dog'):
        numClass = 2
        inSize = 200*200
        NoImages = 0
        for i in range(numClass):
            folderName = args.train + '/'
            if (i==0):
                folderName = folderName + 'cat/'
            else:
                folderName = folderName + 'dog/'
            NoImages += len(os.listdir(folderName)) 
            
        maxImage = 2
        NoImages = maxImage*numClass
        inputs = np.zeros((NoImages,inSize))
        labMat = np.eye(numClass,numClass)
        labels = np.zeros((NoImages,numClass))
        
        itrNo = 0
        for i in range(numClass):
            folderName = args.train + '/'
            
            
            
            imgNo = 0
            if (i==0):
                folderName = folderName + 'cat/'
            else:
                folderName = folderName + 'dog/'
             
            for imgName in os.listdir(folderName):
                
                imgName = folderName+imgName
                
                imgNo += 1
                
                
                if(imgNo<=maxImage):
                    img = util.rgb2gray(mpimg.imread(imgName))
                    
                    inputs[itrNo,:] = img.reshape(inSize)
                    labels[itrNo,:] = labMat[i]
                    itrNo += 1
    
    in_train, in_test, label_train, label_test = train_test_split(inputs, labels, test_size=0.1)
    scaler = StandardScaler()
    scaler.fit(in_train)
    in_train = scaler.transform(in_train)
    in_test = scaler.transform(in_test)
    
    
    if(numNodes[0] != inSize):
        print('Input neurons specified does not match image size')
        exit(0)
    
    if(numNodes[-1] != numClass):
        print('Output neurons specified does not match number of classes')
        exit(0)
    
    
    network = mlp.MLNN()
    network.initializeStruct(numLayers,numNodes,activation)
    network.initializeWeights()
    error, predicted = network.train(in_train,label_train)
    saveFileName = '../Model/' + args.dataset + '.npy'
    network.saveModel(saveFileName)
    
    network.forwardProp(in_train)
    predicted = network.predict(network.nnLayers[numLayers-1].output)
    accuracy = util.getAccuracy(np.argmax(label_train,axis=1),np.argmax(predicted,axis=1))
    f1_macro,f1_micro = util.fi_macro_micro(np.argmax(label_train,axis=1),np.argmax(predicted,axis=1), numClass)
    print('PERFORMANCE ON TRAINING DATA:')
    print(accuracy, f1_macro, f1_micro)
            
    network.forwardProp(in_test)
    predicted = network.predict(network.nnLayers[numLayers-1].output)
    accuracy = util.getAccuracy(np.argmax(label_test,axis=1),np.argmax(predicted,axis=1))
    f1_macro,f1_micro = util.fi_macro_micro(np.argmax(label_test,axis=1),np.argmax(predicted,axis=1), numClass)
    print('PERFORMANCE ON VALIDATION DATA')
    print(accuracy, f1_macro, f1_micro)
    
    
    NoImages = len(os.listdir(args.test)) 
    
    inputs = np.zeros((NoImages,inSize))
    itrNo = 0
    for imgName in os.listdir(args.test):
              
        imgName = args.test+ '/' + imgName
                
        
                
                
        if(args.dataset == 'Cat-Dog'):        
            img = util.rgb2gray(mpimg.imread(imgName))
        else:
            img = mpimg.imread(imgName)
                    
        inputs[itrNo,:] = img.reshape(inSize)
        itrNo += 1
        
    network.forwardProp(inputs)
    predict = network.predict(network.nnLayers[numLayers-1].output)
    predicted = np.argmax(predict,axis=1)
    print(predicted)