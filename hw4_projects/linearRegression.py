#Linear Regression
#Yiming Liu, Yi Ren
#To run this script, please add "linear-regression.txt" in the same path.

import numpy as np
from numpy import *

data = np.loadtxt("linear-regression.txt", delimiter=',')

m,n = data.shape
# x1 represents the constant tern in X matrix [1,x1,x2...]
x1 = ones(m).T
X_o = mat(data[:, :2])
X = np.c_[x1,X_o]
Y = mat(data[:, 2])
Y = Y.T

#apply Normal Equations to compute the weight
weight = np.dot(np.dot(np.dot(X.T,X).I,X.T),Y)
print ("weight =" )
print weight



###############################compute the accuracy#######################
# y_pre = zeros(m)
# for i in range(0,m):
#     y_pre[i] = weight[0]+X_o[i,0]*weight[1]+X_o[i,1]*weight[2]
# u = 0
# v = 0
# for i in range(0,m):
#     u += (Y.item(i)-y_pre[i])*(Y.item(i)-y_pre[i])
#     v += (Y.item(i)-y_pre.mean())*(Y.item(i)-y_pre.mean())
# score = 1.0- u/v
# print score
