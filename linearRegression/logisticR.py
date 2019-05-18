#Logistic Regression
#Yiming Liu, Yi Ren
#To run this script, please add "classification.txt" in the same path.

import numpy as np
from numpy import *
import pandas as pd
data = np.loadtxt("classification.txt",delimiter=',')
m,n = data.shape
x1 = ones(m).T
X_o = data[:,0:3]
X = np.c_[x1,X_o]
Y = data[:,4]

for i in range(0,m):
    Y[i]=(Y[i]+1)/2
Y = Y.reshape(len(data), 1)


# train_x = X[0:999,:]
# train_y = Y[0:999,:]
# test_x = X[1000:1999,:]
# test_y = Y[1000:1999,:]

#sigmoid(logistic) function: 1/(1+e*(-(y*wT*x)))
def sigmoidFunction(x):
    result = 1.0 / (1 + exp(-x))
    return result


def logRe(X,Y):
    #set loop times
    maxTimes = 500
    m,n = X.shape
    weight = np.zeros((n,1))
#initial cost function
    cost = pd.Series(np.arange(maxTimes, dtype= float))
#set learning rate
    rate =  0.001
#use gradient descent to compute best weight
    for i in range(0,maxTimes):
        h = sigmoidFunction(np.dot(X,weight))
        cost[i] = -(1/m)*np.sum(Y*np.log(h)+(1-Y)*np.log(1-h))
        error = h - Y
        gradientDescent = np.dot(X.T, error)
        weight -= rate * gradientDescent
    print ("weight =" )
    print weight
    return weight

weight = logRe(X,Y)


#########test trained Logistic Regression model given test set#############
# def testLogRegres(weights, test_x, test_y):
#     numSamples, numFeatures = shape(test_x)
#     matchCount = 0
#     for i in xrange(numSamples):
#         predict = sigmoidFunction(test_x[i, :] * weights)[0, 0] > 0.5
#         if predict == bool(test_y[i, 0]):
#             matchCount += 1
#     accuracy = float(matchCount) / numSamples
#     return accuracy
#
# weight = logRe(train_x,train_y)
# accuracy = testLogRegres(weight, test_x, test_y)
# print accuracy
