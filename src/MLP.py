#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:14:14 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import random
dataset = 'MNIST'
from sklearn.utils import shuffle


class MLNN:
    def initializeStruct(self, numLayers,numNodes,activation):
        self.nnLayers = []
        self.numLayers = numLayers
        self.numNodes = numNodes
        self.activation = activation
        self.learning_rate = 0.03
        self.batch_size = 100
        self.epochs = 1000
        
    def initializeWeights(self):
        
        
        for i in range(self.numLayers):
            layer = layers(self.numNodes[i])
            if(i!=self.numLayers-1):
                
                #layer.weights = 2*np.random.rand(self.numNodes[i], self.numNodes[i+1]) - 1#np.random.normal(0, 1, size=(self.numNodes[i], self.numNodes[i+1]))
                
                layer.weights = np.random.normal(0, 0.001, size=(self.numNodes[i], self.numNodes[i+1]))
                
                   
            else:
                layer.weights = None
            self.nnLayers.append(layer)
        
    
    def loadWeights(self,fileName):
        self.nnLayers = []
        
        fp = open(fileName,'rb')
        self.numLayers = np.load(fp)
        self.numNodes = np.load(fp)
        
        
        self.activation = np.load(fp)
        print(self.activation)
        for i in range(self.numLayers):
            layer = layers(self.numNodes[i])
            if(i!=self.numLayers-1):
                
                #layer.weights = 2*np.random.rand(self.numNodes[i], self.numNodes[i+1]) - 1#np.random.normal(0, 1, size=(self.numNodes[i], self.numNodes[i+1]))
                
                layer.weights = np.load(fp)
                   
            else:
                layer.weights = None
            self.nnLayers.append(layer)
        
        fp.close()
        return self.numLayers
        
    def train(self,inputs,labels):
        err = np.zeros(self.epochs)
        for j in range(self.epochs):
            i = 0
            inputs[1:,], labels[1:,] = shuffle(inputs[1:,], labels[1:,])
            #inputs, labels = shuffle(inputs, labels)
            
            predicted = np.zeros_like(labels)
            while i+self.batch_size <= len(inputs):
                
                self.error = 0 
                self.forwardProp(inputs[i:i+self.batch_size])
                self.backProp(labels[i:i+self.batch_size])
                predicted[i:i+self.batch_size,:] = self.nnLayers[self.numLayers-1].output
                i += self.batch_size
                
            #if (j == self.epochs-1):
            #    print(predicted)
            predicted = self.predict(predicted)
            #if (j == self.epochs-1):
            #    print(predicted)
            err[j] = self.estimateError(labels,predicted)
        return err, predicted
                  
            
    def forwardProp(self, data):
        self.nnLayers[0].output = data
        for i in range(self.numLayers-1):
            
            out = np.matmul(self.nnLayers[i].output,self.nnLayers[i].weights)
            
            if(i != self.numLayers-2):
                self.nnLayers[i+1].linOutput = out
                self.nnLayers[i+1].linOutput[:,0] = 1
                self.nnLayers[i+1].output = self.activationFunction(out,self.activation)
                
                self.nnLayers[i+1].output[:,0] = 1
                
            else:
                self.nnLayers[i+1].output = self.activationFunction(out,'softmax')
                
    def costFunction(self,costFn,target,predicted):
        if (costFn == 'squared'):
            return target-predicted
        
        
        
          
    def predict(self,output):
        predicted = np.zeros_like(output)
        ind = np.argmax(output,axis=1)
        for i in range(len(output)):
            predicted[i,ind[i]] = 1
        #predicted[np.arange(output), np.argmax(output,1)] = 1
        return predicted
            
    
    def backProp(self,labels):
        
        t = labels
        y = self.nnLayers[self.numLayers-1].output
        
        
        error = self.costFunction('squared',t,y)#t-y
        
        
        
        deriveAct = self.deriveActivationFunction(y,self.nnLayers[self.numLayers-1].linOutput ,'softmax')
        
        self.nnLayers[self.numLayers-1].deltaLocal = np.multiply(error,deriveAct)
        
        
        for i in range(self.numLayers-2,0,-1):
            y = self.nnLayers[i].output
            outSum = self.deriveActivationFunction(y,self.nnLayers[i].linOutput ,self.activation)
            inSum = np.matmul(self.nnLayers[i+1].deltaLocal,self.nnLayers[i].weights.T)
            
            self.nnLayers[i].deltaLocal = np.multiply(outSum,inSum)
            
        for i in range(self.numLayers-1):
            deltaWeight = np.matmul(self.nnLayers[i].output.T,self.nnLayers[i+1].deltaLocal)
            self.nnLayers[i].weights = self.nnLayers[i].weights + self.learning_rate * deltaWeight
            
    def estimateError(self, labels, predicted):
        if len(labels[0]) != self.nnLayers[self.numLayers-1].numNodes:
            print ("Error: Label is not of the same shape as output layer.")
            return
        #for i in range(len(labels)):
         #   print(labels[i],predicted)
        err = np.square(np.subtract(labels, predicted))
        
        self.error = np.mean(err)
        return self.error    
    
    def deriveActivationFunction(self,inData,linOutput, activation):
        
        if(activation == 'tanh'):
            return np.multiply(1+inData,1-inData) 
        if(activation == 'ReLu'):
            der = np.zeros_like(inData)
            ind = np.where(der>0)
            der[ind] = 1
            
            
            return der
        
        if(activation == 'swish'):
            value = np.divide(1, np.add(1, np.exp(-linOutput)))
            return np.add(inData,np.multiply(1-inData, value))
        
        return np.multiply(inData,1-inData) 
            
         
         
    
    def activationFunction(self,inData, activation):
        if(activation == 'sigmoid'):
            return np.divide(1, np.add(1, np.exp(-inData)))
        
        if(activation == 'tanh'):
            
            return np.divide(np.subtract(1, np.exp(-2*inData)), np.add(1, np.exp(-2*inData)))
        
        if(activation == 'ReLu'):
            der = np.zeros_like(inData)
            ind = np.where(der>0)
            der[ind] = 1
            return np.multiply(inData, der)
         
        if(activation == 'swish'):
            value = np.divide(1, np.add(1, np.exp(-inData)))
            return np.multiply(inData, value)
        
        if(activation == 'softmax'):
            value = np.exp(inData)
            denom = np.sum(value, axis=1)
            
            return value/(denom[:,None])
            
    def saveModel(self,fileName):
        fp = open(fileName,'ab')
        np.save(fp,self.numLayers)
        np.save(fp,self.numNodes)
        
        np.save(fp,self.activation)
            
        for i in range(self.numLayers-1):
            np.save(fp,self.nnLayers[i].weights)
        fp.close()
        
        
    
class layers:
    def __init__(self,numNodes):
        self.numNodes = numNodes
        self.output = np.zeros([numNodes,1])
        self.linOutput = np.zeros([numNodes,1])
        self.deltaLocal = np.zeros([numNodes,1])
        
        
def generatePoints(D,nRows,nCols):
    inputs = np.zeros((nRows*nCols,3))
    labels = np.ones((nRows*nCols,1))
    for i in range(nRows):
        for j in range(nCols):
            inputs[i*nCols+j][0] = 1
            inputs[i*nCols+j][1] = i*D-D
            inputs[i*nCols+j][2] = j*D-D
            if((i%2 and (j%2) ==0) or ((i%2) == 0 and j%2)):
                labels[i*nCols+j] = 0
    return inputs, labels
    
        

    
    