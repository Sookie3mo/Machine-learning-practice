
from sklearn import linear_model
from numpy import *
data = loadtxt("linear-regression.txt", delimiter=',')
X = mat(data[:, :2])
Y = mat(data[:, 2])
Y = Y.T
reg = linear_model.LinearRegression()
reg.fit(X, Y)

print ("Predicted Y:")
print(reg.predict(X))
print ("Accuracy:")
print(reg.score(X,Y))


#score(X,Y): defined as (1 - u/v), where u is the regression sum of squares
#  ((y_true - y_pred) ** 2).sum()and v is the residual sum of squares
# ((y_true - y_true.mean()) ** 2).sum().

