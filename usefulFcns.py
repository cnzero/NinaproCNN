import tensorflow as tf
import numpy as np

# Newly build an empty directory
# if exist, delete it and empty it
def BuildNewlyDir(strDirName):
    if tf.gfile.Exists(strDirName):
        print('Existing...')
        try:
            tf.gfile.DeleteRecursively(strDirName)
            print('Successful to delete dirctory')
        except:
            print('Failed to delete directory')
    print('Non-existing...')
    try:
        tf.gfile.MakeDirs(strDirName)
        print('Successful to newly build directory')
    except:
        print('Failed to newly build directory')





# -- Following code for unit test
#a = np.reshape(list(range(1,28)), (3,9))
#print(a)
#c = SplitOnColumn(a, 4)
#print(c)
#
#[[ 1  2  3  4  5  6  7  8  9]
# [10 11 12 13 14 15 16 17 18]
# [19 20 21 22 23 24 25 26 27]]
#
#[[[  1.   2.   3.   4.]
#  [ 10.  11.  12.  13.]
#  [ 19.  20.  21.  22.]]
#
#[[  5.   6.   7.   8.]
# [ 14.  15.  16.  17.]
# [ 23.  24.  25.  26.]]]
#
def SplitOnColumn(array2D, everyNcolumns):
    nRows, nColumns = array2D.shape
    nSlices = int( nColumns/everyNcolumns)
    newArray3D = np.zeros( (nSlices, nRows, everyNcolumns) )
    for i in range(nSlices):
        newArray3D[i, :, :] = array2D[:, everyNcolumns*i:everyNcolumns*(i+1) ]
    
    return newArray3D
