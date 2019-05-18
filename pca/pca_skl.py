import numpy as np
from sklearn.decomposition import PCA

X = np.loadtxt('pca_data.txt')

pca = PCA(n_components = 2)
pca.fit(X)
#print pca.transform(X)
print pca.explained_variance_ratio_
print pca.components_


# [[ 0.86667137 -0.23276482  0.44124968]
#  [-0.4962773  -0.4924792   0.71496368]]
#
# [[ 10.87667009   7.37396173]
#  [-12.68609992  -4.24879151]
#  [  0.43255106   0.26700852]
