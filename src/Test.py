import MLP as mlp
import os
import numpy as np
import util as util
#import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import argparse

def test(args):
    print('Testing of Images in progress...!')
    
    
    

    
    
    activation = 'sigmoid'
    
    if(args.dataset == 'MNIST'):
        inSize = 28*28
    else:
        inSize = 200*200
    
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
        
    
    
    network = mlp.MLNN()
    fileName = '../Model/' + args.dataset + '.npy'
    numLayers = network.loadWeights(fileName)
    network.forwardProp(inputs)
    predict = network.predict(network.nnLayers[numLayers-1].output)
    predicted = np.argmax(predict,axis=1)
    print(predicted)
            
    
    
    #plt.plot(error)
    #network.saveModel('../Model/test.npy')
    
    
    #plt.imshow() 