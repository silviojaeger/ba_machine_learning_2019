import pandas as pd
import numpy as np
import sys
import math
import time
from sklearn.preprocessing import KBinsDiscretizer
from logger import *

# --- Slicer ----------------------------------------------------------------------------------------------------
class Slicer:
    @staticmethod
    def trimHead(dataframe):
        cutName = ''
        safeSet = pd.DateOffset(14)
        cutIndex = pd.Timestamp.min + safeSet
        baseDataFrame = dataframe
        for col in range(dataframe.shape[1]):
            for row in range(dataframe.shape[0]):
                if math.isnan(dataframe.iloc[row,col]):
                    if dataframe.index[row+1]<cutIndex: continue
                    cutIndex = dataframe.index[row+1]
                    cutName = dataframe.columns[col]
                else: 
                    break
            sys.stdout.write(f'\r{Logger.bulk(col, dataframe.shape[1]-1)} Trim Head : {cutIndex} ({cutName})    ')
            dataframe = baseDataFrame.loc[cutIndex - safeSet:] # speed up, including 7 day safety
        dataframe = baseDataFrame.loc[cutIndex:]
        sys.stdout.write('\r\n')
        # fix
        #dataframe = dataframe.fillna(method='ffill') # probably useless
        return dataframe
    
    @staticmethod
    def trimTail(dataframe):
        cutName = ''
        safeSet = pd.DateOffset(14)
        cutIndex = pd.Timestamp.max - safeSet
        baseDataFrame = dataframe
        for col in range(dataframe.shape[1]):
            for row in range(dataframe.shape[0]-1, 0-1, -1):
                if math.isnan(dataframe.iloc[row,col]):
                    if dataframe.index[row]>cutIndex: continue
                    cutIndex = dataframe.index[row-1]
                    cutName = dataframe.columns[col]
                else: 
                    break
            sys.stdout.write(f'\r{Logger.bulk(col, dataframe.shape[1]-1)} Trim Tail : {cutIndex} ({cutName})    ')
            dataframe = baseDataFrame.loc[:cutIndex + safeSet] # speed up, including 7 day safety
        dataframe = baseDataFrame.loc[:cutIndex]
        sys.stdout.write('\r\n')
        # fix
        #dataframe = dataframe.fillna(method='bfill') # probably useless
        return dataframe
        
    @staticmethod
    def split(dataframe, trainsetSize=0.8):
        """ split series into two parts
            usage: trainData, testData = split(series)
        """
        rows      = dataframe.shape[0]  # gives number of row count
        trainSize = int(rows*trainsetSize)

        train = dataframe[:trainSize]
        test  = dataframe[trainSize:]
        return (train, test)

    #binser = KBinsDiscretizer(n_bins=numCategories, encode='onehot', strategy='uniform')
    #binser.fit(yDiff)
    #yCat = binser.transform(yDiff)
    @staticmethod
    def sliceCategory(dataframe, block=5, predictionLength=1, numCategories=5):
        """ slice data into block and prediction
            usage: block, prediction = slice(series)
        """
        if numCategories%2 == 0: raise("numCategories has to be an odd number")
        progress = 0
        length = dataframe.shape[0]-block-predictionLength
        x = [[[0.0]*dataframe.shape[1] for i in range(block)] for j in range(length)]
        y = [0.0]*length
        start = time.time()
        for i in range(length):
            progress += 1
            if (i%1000 == 0 and i > 0) or i+1 == length: 
                duration = time.time() - start
                if duration == 0: duration = 0.00001
                start = time.time()
                sys.stdout.write(f'\r{Logger.bulk(progress, length)} {1000/duration:.2f} records/s, {length/(1000/duration)/60:.2f}min left')
            valX = dataframe.iloc[i       : i+block           , :] # actual block eg last week
            
            val1 = dataframe.iloc[i+block, 0] # last block value for diff
            val2 = dataframe.iloc[i+block+predictionLength, 0] # last block value + prediction for diff
            valDiff = val2-val1 # dy / dx ==> dx = 1
            # capsule
            x[i] = valX.values
            y[i] = valDiff
        sys.stdout.write('\r\n') 

        # gauss borders
        yAbs = [abs(val) for val in y]
        yAbs.sort()
        # biggestDiff = yAbs[int(0.75*length)]
        biggestDiff = yAbs[int(0.75*length)]

        #---Classification----------------------------------------------------------------------------
        classSize = biggestDiff*2/numCategories
        for i in range(length):
            classFound = False
            classArray = np.zeros(numCategories)
            # check (from top to bottom) if value fits to the class
            # Class '0' is biggest diff upwards, '4' is biggest downwards, '2' is flat (whith 5 classes)
            for clas in range(numCategories):
                if (y[i] >= (biggestDiff-((clas+1)*classSize))):
                    classArray[clas] = 1
                    classFound = True
                    break
            #if no class fits, the value is the most negative of all time so it fits to the last class
            if not classFound:
                classArray[numCategories-1] = 1
            #set classArray as label
            y[i]=classArray
        #----------------------------------------------------------------------------------------------
        x = np.array(x)
        y = np.array(y)
        return x, y

    @staticmethod
    def sliceMixed(dataframe, block=5, prediction=1, specialTimeWindows=[]):
        """ slice data into block and prediction
            usage: block, prediction = slice(series)
        """
        # slice
        progress = 0
        length = dataframe.shape[0]-block-prediction
        x = [[[0.0]*dataframe.shape[1] for i in range(block)] for j in range(length)]
        y = [[0.0]*prediction for i in range(length)]
        start = time.time()
        for i in range(length):
            progress += 1
            if (i%1000 == 0 and i > 0) or i+1 == length: 
                duration = time.time() - start
                if duration == 0: duration = 0.00001
                start = time.time()
                sys.stdout.write(f'\r{Logger.bulk(progress, length)} {1000/duration:.2f} records/s, {(length-i)/(1000/duration)/60:.2f}min left')
            valX = dataframe.iloc[i:i+block, :].values # actual block eg last week
            for window in specialTimeWindows:
                if i-block*window > 0: valX = np.append([dataframe.iloc[i-block*window, :].values], valX, axis=0)
                else: valX = np.append([dataframe.iloc[0, :].values], valX, axis=0)
            valY = dataframe.iloc[i+block-1+prediction, 0] # the result 
            pVal = dataframe.iloc[i+block-1, 0]
            # capsule
            x[i] = valX
            y[i] = [valY, pVal]
        sys.stdout.write('\r\n')
        x = np.array(x)
        y = np.array(y)
        return x, y

    @staticmethod
    def slice(dataframe, block=5, prediction=1):
        """ slice data into block and prediction
            usage: block, prediction = slice(series)
        """
        # slice
        progress = 0
        length = dataframe.shape[0]-block-prediction
        x = [[[0.0]*dataframe.shape[1] for i in range(block)] for j in range(length)]
        y = [[0.0]*prediction for i in range(length)]
        start = time.time()
        for i in range(length):
            progress += 1
            if (i%1000 == 0 and i > 0) or i+1 == length: 
                duration = time.time() - start
                if duration == 0: duration = 0.00001
                start = time.time()
                sys.stdout.write(f'\r{Logger.bulk(progress, length)} {1000/duration:.2f} records/s, {length/(1000/duration)/60:.2f}min left')
            valX = dataframe.iloc[i       : i+block           , :] # actual block eg last week
            valY = dataframe.iloc[i+block+prediction, 0] # the result 
            # capsule
            x[i] = valX.values
            y[i] = valY
        sys.stdout.write('\r\n')
        x = np.array(x)
        y = np.array(y)
        return x, y
# ---------------------------------------------------------------------------------------------------------------