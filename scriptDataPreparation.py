from DataPreparation import *
import numpy as np
import pickle
import time


timeStart = time.time()
rawDict = getRawDict()
(Images, Labels) = getRmsImagesLabels(rawDict)
print('All clusters Images shape: ', Images.shape)
print('All clusters Labels shape: ', Labels.shape)
timeEnd = time.time()
print('Extracting RMS feature from .mat file needs time:\n', timeEnd-timeStart)

timeStart = time.time()
ninaRmsImagesLabels = {}
ninaRmsImagesLabels['Images'] = Images
ninaRmsImagesLabels['Labels'] = Labels
# write to 'ninaRmsImagesLabelsFiles.pkl' with pickle
pickle.dump(ninaRmsImagesLabels, open('ninaRmsImagesLabelsFiles.pkl', 'wb'))
timeEnd = time.time()
print('Writing ImagesLabels to .pkl file needs time:\n', timeEnd - timeStart)


# read from 'ninaRmsImagesLabelsFiles.pkl' for checking
timeStart = time.time()
newImagesLabels = pickle.load(open('ninaRmsImagesLabelsFiles.pkl', 'rb'))
timeEnd = time.time()
print('Reading ImagesLabels from .pkl file needs time:\n', timeEnd - timeStart)

assert(np.all( newImagesLabels['Images']==Images)), 'something wrong with Images value'
assert(np.all( newImagesLabels['Labels']==Labels)), 'something wrong with Labels value'


