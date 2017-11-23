import numpy as np

def fRMS(d):
    return np.sqrt( np.mean( np.square(d), axis=0, keepdims=True ) )

def fMAV(d):
    return np.mean(np.abs(d), axis=0, keepdims=True)

def fZC(d):
    d = np.array(d)
    nZC = 0
    th = np.mean(d, axis=0, keepdims=True)
    th = abs(th)
    for i in range(1, d.shape[0]):
        if d[i-1]<th and d[i]>th:
            nZC += 1
        elif d[i-1]>th and d[i]<th:
            nZC += 1

    return nZC/d.shape[0]

def fSSC(d):
    d = np.array(d)
    nSSC = 0;
    th = np.mean(d, axis=0, keepdims=True)
    th = abs(th)
    for i in range(2, d.shape[0]):
        diff1 = d[i] - d[i-1]
        diff2 = d[i-1]-d[i-2]
        if abs(diff1)>th and abs(diff2)>th and (diff1*diff2)<0:
            nSSC += 1
    return nSSC/d.shape[0]

def fVAR(d):
    return np.var(d, axis=0)

# Function descriptioin:
#    Features Extraction with Sliding Window

# Input parameters:
#    data, LxnCh, emg raw data of sequence
#    feStr, function handles of features extraction functions
#           actually, it is list with each element is the features name in string, `f` is only difference with its feature extraction name. 
#    LI, scale, length of incremental window, for example 30ms = 60/2000
#    LW, scale, length of extraction window, for example 100ms = 200/2000
# Output parameters:
#    fe, 2-D matrix, nW x (nFe x nCh)
def extractSlidingWindow(data,feStr, LI, LW):
    Row, Column = data.shape
    nW = int( (Row - LW)/LI)
    nFE = feStr.shape[0]
    # print(Row, Column, nW, nFE)
    
    feMatrix = np.zeros([nW, nFE*Column])
    for w in range(nW):
        feRow = np.zeros([1, nFE*Column])
        for c in range(Column):
            dataWindow = data[w*LI:w*LI+LW, c]
            for nf in range(nFE):
                festr = feStr[nf, 0]
                fe = eval( 'f'+festr+'('+str(list(dataWindow) )+')' )
                feRow[0, nFE*c+nf] = fe
        feMatrix[w, :] = feRow[0, :]
    return feMatrix
