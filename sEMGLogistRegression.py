from classNinapro import Ninapro
import numpy as np
from usefulFcns import *


ninapro = Ninapro()
ninapro.splitImagesLabels()

# Train
print('ninapro.TrainImages shape: ', ninapro.TrainImages.shape)   # m x 16 x 30
print('ninapro.TrainLabels shape: ',  ninapro.TrainLabels.shape)  # m x 8
# Test
print('ninapro.TestImages shape: ', ninapro.TestImages.shape)     # m x 16 x 30
print('ninapro.TestLabels shape: ', ninapro.TestLabels.shape)     # m x 8
# Validate
print('ninapro.ValidateImages shape: ', ninapro.ValidateImages.shape) # m x 16 x 30
print('ninapro.ValidateLabels shape: ', ninapro.ValidateLabels.shape) # m x 8

print('Read successfully  done...')
# Scale the original RMS pixel value
#ninapro.TrainImages *= 1000
#ninapro.TestImages *= 1000
#ninapro.ValidateImages *= 1000

# number of total classes of movements, 8 for exampel.
nMV = ninapro.TrainLabels.shape[1]
partIndex = [0,1,2,3,4,5,6,10,11,12,13,14,15] # exclude [7,8,9] these three channels.
nCh = 13

# -- Logistic Regression
from sklearn import linear_model
xTrain = np.reshape(ninapro.TrainImages[:, partIndex, :], (-1, nCh*30))
yTrain = np.argmax(ninapro.TrainLabels, axis=1)
print('xTrain shape: ', xTrain.shape)
print('yTrain shape: ', yTrain.shape)

xTest = np.reshape(ninapro.TestImages[:, partIndex, :], (-1, nCh*30))
yTest = np.argmax(ninapro.TestLabels, axis=1)

cBackup = [1, 1e1, 1e3, 1e5, 1e7, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14,1e42]

for c in cBackup:
    print(c)
    logreg = linear_model.LogisticRegression(C=c, penalty='l2')
    logreg.fit(xTrain, yTrain)
    accTrain = np.mean(yTrain==logreg.predict(xTrain))
    print('Train accuracy: ', accTrain)

    yhatTest = logreg.predict(xTest)
    accTest = np.mean(yTest==yhatTest)
    print('Test accuracy: ', accTest)

    print(yTest[0:30])
    print(yhatTest[0:30])
