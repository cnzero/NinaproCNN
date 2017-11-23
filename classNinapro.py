import numpy as np
import pickle

class Ninapro(object):
    def __init__(self, dataPath='ninaRmsImagesLabelsFiles.pkl'):
        readImagesLabels = pickle.load(open(dataPath, 'rb'))
        self.Images = readImagesLabels['Images']
        # 3D numpy array
        # mx16x30

        self.Labels = readImagesLabels['Labels']
        # 2D numpy array
        # mx8

        # next_batch indexing
        self.nextBatchIndex = 0
        
    def splitImagesLabels(self, proportion=[0.7, 0.2, 0.1]):
        # shuttle or random firstly. 
        nSample = self.Images.shape[0]
        sampleShuffleIndex = list(range(0, nSample))
        np.random.shuffle(sampleShuffleIndex) 

        nP = [int(p*nSample) for p in proportion]
                
        # - Train
        self.TrainImages = self.Images[ sampleShuffleIndex[0:nP[0] ], :, :]
        self.TrainLabels = self.Labels[ sampleShuffleIndex[0:nP[0] ], :]
        # - Test
        self.TestImages = self.Images[ sampleShuffleIndex[nP[0]:nP[0]+nP[1]], :, :]
        self.TestLabels = self.Labels[ sampleShuffleIndex[nP[0]:nP[0]+nP[1]], :]
        # - Validate
        self.ValidateImages = self.Images[ sampleShuffleIndex[nP[0]+nP[1]:-1], :, :]
        self.ValidateLabels = self.Labels[ sampleShuffleIndex[nP[0]+nP[1]:-1], :]

    def next_batch(self, nextN):
        m = self.TrainImages.shape[0]
        if (self.nextBatchIndex + nextN) < m:
            sliceIndex = list(range(self.nextBatchIndex, self.nextBatchIndex+nextN))
            # update
            self.nextBatchIndex += nextN

        elif (self.nextBatchIndex < m) and (self.nextBatchIndex + nextN) > m:
            # not too much samples for this batch
            sliceIndex = list(range(self.nextBatchIndex, m))
            # update
            self.nextBatchIndex += nextN

        else: # self.nextBatchIndex == m
            # update
            self.nextBatchIndex = 0
            sliceIndex = list(range(0, nextN))
        
        return self.TrainImages[sliceIndex, :, :], self.TrainLabels[sliceIndex, :]  # ImagesBatch, LabelsBatch

