import numpy as np
from scipy import io
from FeaturesExtraction import extractSlidingWindow
from usefulFcns import SplitOnColumn

def getRawDict(matPath='DB6_s1_a_DB6_s1_a_S1_D1_T1.mat'):
    mat = io.loadmat(matPath)
    L, nChEMG = mat['emg'].shape
                            # L = 1428729, 
                            # nChEMG = 16,
    nChACC = mat['acc'].shape[1]
                            # nChACC = 48
    movement = np.reshape( np.unique(mat['restimulus']), (1, -1) )
                            # movement = [ 0  1  3  4  6  9 10 11] 强制为1x8
    nMV = movement.shape[1]
                            # nMV = 8
    emg = mat['emg'][:, :]
    restimulus = mat['restimulus'][:, :]
    
    # Preparation for discarding head and tail
    restimulus_shift1 = np.append(restimulus[1:], [0])                   # restimulus.shape Lx1
    restimulus_shift1 = np.reshape(restimulus_shift1, (-1, 1))           # restimulus_shift1.shape Lx1
    diff_restimulus = restimulus_shift1 - restimulus
    cutlineIndex = diff_restimulus != 0                                  # True-value, shape Lx1
    indexscale = np.reshape( np.linspace(1, L, L, dtype=np.int), (-1, 1) ) # axis integers index scale column, 1:L, 
    
    cutlinePosition = np.reshape(indexscale[cutlineIndex], (-1, 1) )     # positions of cutting line  168x1
                                                                        # Explanations: 12repetiion X 7movements X 2 = 168
    discardLength = 1000 # 1000/2KHz = 500ms
    discardIndexValue = np.reshape(np.zeros(L, dtype=np.int), (-1, 1))
    
    for axi in cutlinePosition[:, 0]:
        discardIndexValue[axi-discardLength : axi+discardLength, 0] = 20
        
    mvIndex = restimulus - discardIndexValue
    
    ### Use dictionary to story corresponding emg data
    label_rawDataD = {}
    # label as the [keys] of the dictionary
    # rawData of that label as the  [values] of the dictionary
    
    for i in range(nMV):
        condition = mvIndex == movement[0, i]
        condition = np.repeat(condition, nChEMG, axis=1)
        rawData = emg[condition]
        rawData = np.reshape(rawData, (-1, nChEMG))
        # print(str(i)+' movement raw data shape: '+str(rawData.shape))  # print to debug
        label_rawDataD[str(i)] = rawData
    
    return label_rawDataD


def getRmsImagesLabels(rawDict, rmsFE=np.array([['RMS']]), LI=8, LW=10):
    clusterMatrix2D = np.transpose(extractSlidingWindow(rawDict[str(0)], rmsFE, LI, LW))
    clusterMatrix3D = SplitOnColumn(clusterMatrix2D, 30)
    m = clusterMatrix3D.shape[0]
    nMV = len(rawDict.keys() )
    labelOneHot = np.zeros( (m, nMV) )
    labelOneHot[:, 0] = 1
    # append to Images-Labels pair
    Images = clusterMatrix3D
    Labels = labelOneHot
 
    for i in range(1, 8):
        clusterMatrix2D = np.transpose(extractSlidingWindow(rawDict[str(i)], rmsFE, LI, LW))
        # every key->value, shape [16 x Lfi]
        clusterMatrix3D = SplitOnColumn(clusterMatrix2D, 30)
        m = clusterMatrix3D.shape[0] # m image[16x30] for such a cluster
        labelOneHot = np.zeros( (m,nMV) )
        labelOneHot[:, i] = 1
        Images = np.append(Images, clusterMatrix3D, axis=0) # more rows with the same columns
        Labels = np.append(Labels, labelOneHot, axis=0) # more rows with the same columns

    return Images, Labels
