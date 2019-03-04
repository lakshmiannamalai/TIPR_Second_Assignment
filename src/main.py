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
import TrainAndTest as train
import Test as test
if __name__ == '__main__':
    print('Welcome to the world of neural networks!')
    
    import argparse
    
    parser = argparse.ArgumentParser(description='MLP')
    parser.add_argument('--train', required=False,
                        help='path to train data')
    parser.add_argument('--test', required=True,
                        help='path to test data')
    parser.add_argument('--dataset', required=True,
                        help='dataset name: MNIST / Cat-Dog')
    parser.add_argument('--configuration', required=False,
                        help='[Nl_1 Nl_2 Nl_3]. Length of the vector is equal to number of layers')
    args = parser.parse_args()
    
    
    if(args.train == None):
        test.test(args)
    else:
        train.trainAndtest(args)
    #dataset = 'MNIST'
    #train = '../data/MNIST/'
    
    
    
    