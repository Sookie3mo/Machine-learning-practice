from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
data = np.loadtxt("classification.txt",delimiter=',')
X = data[:,0:3]
Y = data[:,4]
m = LogisticRegression(max_iter=1000,tol=0.0000001)
m.fit(X,Y)
expected = Y
predicted = m.predict(X)
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))
print(m.score(X,Y))
print(m.predict_proba(X))
print(m.decision_function(X))


