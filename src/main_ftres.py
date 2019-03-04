import MLP as mlp
import os
import numpy as np
import util as util
#import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#import MPLkeras as MLPkeras
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
import BagOfWords as BW
import csv
if __name__ == '__main__':
    print('Welcome to the world of neural networks!')
    
    import argparse
    
#    parser = argparse.ArgumentParser(description='MLP')
#    parser.add_argument('--train', required=True,
#                        help='path to train data')
#    parser.add_argument('--test', required=True,
#                        help='path to test data')
#    parser.add_argument('--dataset', required=True,
#                        help='dataset name: MNIST / Cat-Dog')
#    parser.add_argument('--configuration', required=True,
#                        help='[No of neurons in l1, No of neurons in l2, ...]. Length of the vector is equal to number of layers')
#    args = parser.parse_args()
    
    
    dataFile = '../data/twitter/twitter.txt'
    labelFile = '../data/twitter/twitter_label.txt'
    dataset = 'twitter'
    train = '../data/MNIST/'
    
    
    if (dataset == 'dolphins'):
        numClass = 4
    if (dataset == 'pubmed'):
        numClass = 3
    if (dataset == 'twitter'):
            numClass = 3
    if dataset == 'twitter':
        labelInt = []
        i = 0
        with open(labelFile) as file:
            
            for line in file:
                
                value = line#file.readline().split()
        
                labelInt.append(int(value))   
        if dataFile.find('.txt') == -1:
            with open(dataFile) as csv_file:
                file_id = csv.reader(csv_file, delimiter=',')
                index = 0
                feature = []
                for line in file_id:
                    feature.append(line)
                    index += 1     
        
        
        
            
        
            rowIndex = 0
            Initialize = 1
            for row in feature:
                for i in row:
                    x = i.split(" ")
                    if Initialize == 1:
                        featureDim = len(x)
                        featureInt = np.zeros((len(feature),featureDim))
                        Initialize = 0
                    for colIndex in range(len(x)):
                        featureInt[rowIndex,colIndex] = x[colIndex]
                    
                rowIndex += 1
        else:    
            [featureInt,labelInt,featureDim] = BW.BoW(labelFile,dataFile)
        
    else:
        
        with open(labelFile) as csv_file:
            
            file_id = csv.reader(csv_file, delimiter=';')
            index = 0
            label = []
            for line in file_id:
                label.append(line)
                index += 1
            labelInt = [int(i[0]) for i in label]   
            
                
                
        
        with open(dataFile) as csv_file:
            file_id = csv.reader(csv_file, delimiter=',')
            index = 0
            feature = []
            for line in file_id:
                feature.append(line)
                index += 1     
        
        
        
            
        
        rowIndex = 0
        Initialize = 1
        for row in feature:
            for i in row:
                x = i.split(" ")
                if Initialize == 1:
                    featureDim = len(x)
                    featureInt = np.zeros((len(feature),featureDim))
                    Initialize = 0
                for colIndex in range(len(x)):
                    featureInt[rowIndex,colIndex] = x[colIndex]
                
            rowIndex += 1
    
    
    
    in_train, in_test, label_train, label_test = train_test_split(featureInt, labelInt, test_size=0.1)
    scaler = StandardScaler()
    scaler.fit(in_train)
    in_train = scaler.transform(in_train)
    in_test = scaler.transform(in_test)
    
    
    numLayers = 3
    numNodes = [featureDim,10,numClass]
    activation = 'sigmoid'
    
    #MLPkeras.MLPkeras(in_train,label_train,in_test,label_test,numNodes,activation)
    label_train_in = np.zeros((len(label_train),numClass))
    #label_orig = np.ones(3,3)
    for i in range(len(label_train)):
        label_train_in[i,label_train[i]] = 1
        
    label_test_in = np.zeros((len(label_test),numClass))
    
    for i in range(len(label_test)):
        label_test_in[i,label_test[i]] = 1
    
    network = mlp.MLNN()
    network.initializeStruct(numLayers,numNodes,activation)
    network.initializeWeights()
    error, predicted = network.train(in_train,label_train_in)
    #saveFileName = '../Model/' + dataset + '_' + '{}'.format(numLayers) + '_' + '{}'.format(numNodes[1]) + '.npy'
    #network.saveModel(saveFileName)
    
    network.forwardProp(in_train)
    predicted = network.predict(network.nnLayers[numLayers-1].output)
    accuracy = util.getAccuracy(np.argmax(label_train_in,axis=1),np.argmax(predicted,axis=1))
    f1_macro,f1_micro = util.fi_macro_micro(np.argmax(label_train_in,axis=1),np.argmax(predicted,axis=1), numClass)
    print(accuracy, f1_macro, f1_micro)
            
    network.forwardProp(in_test)
    predicted = network.predict(network.nnLayers[numLayers-1].output)
    accuracy = util.getAccuracy(np.argmax(label_test_in,axis=1),np.argmax(predicted,axis=1))
    f1_macro,f1_micro = util.fi_macro_micro(np.argmax(label_test_in,axis=1),np.argmax(predicted,axis=1), numClass)
    print(accuracy, f1_macro, f1_micro)
    
    
    
#    network1 = mlp.MLNN()
#    
#    numLayers = network1.loadWeights('../Model/MNIST_3_10.npy')
#    network1.forwardProp(inputs)
#    predicted = network1.predict(network.nnLayers[numLayers-1].output)
#            
#    
#    accuracy = util.getAccuracy(np.argmax(labels,axis=1),np.argmax(predicted,axis=1))
#    print(accuracy)
    
    
    #plt.imshow() 