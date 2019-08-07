import sys
import os
import pandas as pd
import numpy as np
import re
from logger import *
from ta import *


rigthClassPerc = 0
rigthDirectionPerc = 0
bestDirectionPerc = 0

directionVerySurePerc = 0
bestDirectionVerySurePerc = 0

# --- Calculator ------------------------------------------------------------------------------------------------
class Calculator:
    
    @staticmethod
    def parseFunctions(df, functions):
        
        functionsCpy = functions.copy()
        for fsym in functionsCpy:
            # interpreter
            sym = re.sub(r"_.*", '', fsym)
            params= re.sub(r"[A-Z,a-z,0-9]+_", '', fsym)
            key = re.sub(r"[0-9]+", '', params)
            arg = re.sub(r"[A-Z,a-z]*", '', params)
            # condition
            if (key == None or arg == None): continue
            # calculation
            if (key == "MA"): 
                df[fsym] = df[sym].rolling(window=int(arg)).mean()
            if (key == "EMA"): 
                df[fsym] = df[sym].ewm(span=int(arg),adjust=False).mean()
            if (key == "BB"): 
                functions.remove(fsym)
                functions.append(fsym+'U')
                functions.append(fsym+'L')
                ma  = df[sym].rolling(window=int(arg)).mean()    
                std = df[sym].rolling(window=int(arg)).std()   
                df[fsym+'U'] = ma + (std)
                df[fsym+'L'] = ma - (std)
            if (key == "SMA"):
                df[fsym] = df[sym].rolling(window=int(arg)).mean().shift(periods=-int(int(arg)/2))
        return df

    @staticmethod
    def crossCorrelationDf(df, targetCompany, lag=5, time=20, method='spearman', showProgress=True):
        if df.shape[1] <= 1: return
        corrList = []
        progress = 0
        for company in df:
            progress += 1
            if showProgress: sys.stdout.write(f'\r{Logger.bulk(progress, df.shape[1])} {targetCompany}    ')
            corrList += [Calculator.crossCorrelation(df[targetCompany], df[company], lag=lag, time=time, method=method)]
        if showProgress: sys.stdout.write('\r\n')
        cross = pd.DataFrame(
                data =  corrList,
                index = [s for s in df],
                columns = ['shift', 'index'])
        # evaluate
        cross['index'] = cross['index'].abs() # build abs
        cross = cross[cross.index != targetCompany] # delete self correlation
        return cross

    @staticmethod
    def crossCorrelatingDataframes(df, targetCompany, count=-1, lag=5, time=20, method='spearman'):
        cross = crossCorrelationDf(df, targetCompany, count=-1, lag=5, time=20, method='spearman')
        cross = cross[cross['shift'] > 0] # select all positive shifts
        cross = cross.sort_values('index', ascending=False) # sort desc
        if count > 0: cross = cross[:count] # get top correlations as defined in config
        print(f'Correlation Result for {targetCompany}')
        print(cross)
        targetSymbols = cross.index
        return df[targetSymbols]

    @staticmethod
    def crossCorrelation(sr1, sr2, lag=5, time=20, method='spearman'):
        """ create cross correlation of two Panda Series
            usage: result = crossCorrelation(series1, series2)

            methods: pearson, kendall, spearman, callable
        """
        # pearson : standard correlation coefficient
        # kendall : Kendall Tau correlation coefficient
        # spearman : Spearman rank correlation
        # callable: callable with input two 1d ndarray
        #
        # Peason vs Spearman:
        # Pearson benchmarks linear relationship, Spearman benchmarks monotonic relationship (few infinities more general case, 
        # but for some power tradeoff).
        #
        # So if you assume/think that the relation is linear (or, as a special case, that those are a two measures of the same 
        # thing, so the relation is y=1⋅x+0) and the situation is not too weired (check other answers for details), go with 
        # Pearson. Otherwise use Spearman.
        timespan = time + 2*lag
        maxIndex = 0
        bestShift = 0
        sr1 = sr1[-timespan:]
        sr2 = sr2[-timespan:]
        for days in range(-lag, lag+1, 1):
            index = sr1[lag:lag+time].corr(sr2.shift(periods=days)[lag:lag+time], method=method)
            if abs(index) > abs(maxIndex): 
                maxIndex = index
                bestShift = days
        return [bestShift, maxIndex]
    
    @staticmethod
    def logReturn(df):  
        df = df.copy()
        if isinstance(df, pd.DataFrame):  
            for column in df:
                df[column] = Calculator.logReturn(df[column])
            df = df.fillna(0.)
            return df
        # fnc for Series
        sr = df
        sr = np.log(sr / sr.shift(1))
        sr = sr.replace([np.inf, -np.inf], 0.)
        sr = sr.fillna(0.)
        return sr

    @staticmethod
    def boolean(df, border):
        if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
            df = df.copy()
            if isinstance(df, pd.DataFrame):  
                for column in df:
                    df[column] = Calculator.boolean(df[column], border)
                return df
            # fnc for Series
            sr = df
            sr = pd.Series([(True if v>=border else False) for v in sr.values], index=sr.index)
            return sr
        if isinstance(df, np.ndarray):
            array = np.empty(df.shape)
            for ele in range(df.shape[0]):
                if isinstance(df[ele], np.ndarray):
                    array[ele] = Calculator.boolean(df[ele], border)
                else:
                    array[ele] = (True if df[ele]>=border else False)
            return array

    @staticmethod
    def scale(df):
        df = df.copy()
        if isinstance(df, pd.DataFrame):  
            for column in df:
                df[column] = Calculator.scale(df[column])
            return df
        # fnc for Series
        sr = df
        if sr.dtype == np.dtype('bool'): return
        maxValue = np.nanmax(sr.values)
        minValue = np.nanmin(sr.values)
        if abs(minValue) > maxValue: maxValue = abs(minValue)
        sr = pd.Series([v/maxValue for v in sr.values], index=sr.index)
        #sr = sr.fillna(0.) # fill empty values
        return sr

    @staticmethod
    def scaleRelative(df):
        df = df.copy()
        df = df / df.max().max() * 0.8
        #df = df.fillna(0.) # fill empty values
        return df

    # @staticmethod
    # def ownMetrik(y_true, y_pred):
    #     length = y_true.shape[0]
    #     confMatrix = [[0.0]*length]*length
    #     index = list(y_true).index(1)
        
    #     for i in range(length):
    #         confMatrix[index][i] += y_pred[i]

    #     logDirTensorboard = os.path.join(os.getcwd(), "logtest")
    #     cm = confusion_matrix(y_true, y_pred)
    #     return True

    @staticmethod
    def compareMatrix(pred, truth):
        length = truth.shape[1]
        # build matrix
        confMatrix = [[0.0]*length for i in range(length)]
        # fill matrix
        highest = 0
        for sample in range(truth.shape[0]):
            index = list(truth[sample]).index(1)
            for i in range(length):
                confMatrix[index][i] += pred[sample][i]
        return confMatrix

    @staticmethod
    def checkMoneyMakerMixed(pred, truth, predictionLength):
        # get 20% of smallest diffs from all stock changes
        diff = []
        for i in range(len(pred)-predictionLength):
            singleDiff = truth[i+predictionLength] - truth[i]
            diff += [abs(singleDiff)]
        diff.sort()
        cutPart = int(0.8*len(diff))
        cut = diff[:-cutPart]
        # pick boarder for classification
        maxInMiddle = max(cut)
        # winning percentage
        winsGradient = 0
        winsDirection = 0
        validDirections = 0
        for i in range(len(pred)-predictionLength):
            predDir      = (pred[i+predictionLength]  - pred[i])
            realPredDir  = (pred[i+predictionLength]  - truth[i])
            truthDir     = (truth[i+predictionLength] - truth[i])
            if abs(truthDir) > maxInMiddle:
                validDirections += 1
                if (predDir>0)==(truthDir>0):
                    winsGradient += 1
                if (realPredDir>0)==(truthDir>0):
                    winsDirection += 1
        winPercentageGradient  = winsGradient /validDirections*100
        winPercentageDirection = winsDirection/validDirections*100
        return winPercentageGradient, winPercentageDirection

    @staticmethod
    def checkMoneyMaker(pred, truth, predictionLength):
        # winning percentage
        winsGradient = 0
        winsDirection = 0
        validDirections = 0
        for i in range(len(pred)-predictionLength):
            predDir      = (pred[i+predictionLength]  - pred[i])
            realPredDir  = (pred[i+predictionLength]  - truth[i])
            truthDir     = (truth[i+predictionLength] - truth[i])
            if truthDir == 0: continue
            validDirections += 1
            if (predDir>0)==(truthDir>0):
                winsGradient += 1
            if (realPredDir>0)==(truthDir>0):
                winsDirection += 1
        winPercentageGradient  = winsGradient /validDirections*100
        winPercentageDirection = winsDirection/validDirections*100
        return winPercentageGradient, winPercentageDirection

    @staticmethod
    def checkMoneyMakerClassification(pred, truth, checkOverValue): 
        #---------------debug-----------
        print(f"Pred: {pred[-1:]}")
        #--------------------------------

        rigthClass = 0

        rightDirection = 0       
        nDirections = 0

        rightDirectionVerySure = 0 
        nDirectionVerySure = 0

        length = truth.shape[1]
        middleClass = int((length-1)/2)
       
        for sample in range(truth.shape[0]):
            index = list(truth[sample]).index(1)
            highest = 0
            indexHighest = 0

            #set indexHighest and highest value from highest prediction
            for i in range(length):
                if (pred[sample][i]>highest): 
                    highest=pred[sample][i]
                    indexHighest=i

            #check if right class was predictet
            if (index == indexHighest):
                rigthClass += 1

            #Check directions
            if ((index>middleClass and indexHighest>middleClass) or (index<middleClass and indexHighest<middleClass)):
                rightDirection+=1
            if (index != middleClass):
                nDirections += 1

            #Check direction with prediction over a certen value
            if(highest >= checkOverValue):
                nDirectionVerySure += 1
                if ((index>middleClass and indexHighest>middleClass) or (index<middleClass and indexHighest<middleClass)):
                    rightDirectionVerySure += 1
        
        #calculate %
        global rigthClassPerc, rigthDirectionPerc, bestDirectionPerc, directionVerySurePerc, bestDirectionVerySurePerc
        rigthClassPerc = rigthClass/truth.shape[0]*100
        if(nDirections>0): rigthDirectionPerc = rightDirection/nDirections*100
        else: rigthDirectionPerc = 0
        if(rigthDirectionPerc>bestDirectionPerc): bestDirectionPerc = rigthDirectionPerc

        if(nDirectionVerySure>0): directionVerySurePerc = rightDirectionVerySure/nDirectionVerySure*100
        else: directionVerySurePerc = 0
        if(directionVerySurePerc>bestDirectionVerySurePerc): bestDirectionVerySurePerc = directionVerySurePerc

        #Print
        print(f'Right class predicted: {rigthClassPerc} %')
        print(f'Right direction predicted: {rigthDirectionPerc} %')
        print(f'Best direction prediction: {bestDirectionPerc} %')
        print(f'Right direction predicted with sureness over {checkOverValue}: {directionVerySurePerc} %')
        print(f'Best direction predicted with sureness over {checkOverValue}: {bestDirectionVerySurePerc} %')

    @staticmethod
    def shiftY(x, y):
        for i in range(len(y)):
            y[i] = y[i] - x[i,-1,0] 
        #dataset = dataset / np.max(dataset)
        #return (x, y)

    @staticmethod
    def reshiftY(x, y):
        for i in range(len(y)):
            y[i] = y[i] + x[i,-1,0]

#---Custom Loss Functions---------------------------------------------------------------------------------------
    @staticmethod
    def customLoss(conf):
        from keras.losses import mse, mae
        from keras.backend import mean, print_tensor, get_value, shape
        from tensorflow.math import atan, tanh, subtract
        from tensorflow import constant
        absoluteMass = conf['absoluteMass']
        gradientMass = conf['gradientMass']
        def loss(y_pred, y_true):
            #y_true = print_tensor(y_true, message='y_true = ')
            #y_pred = print_tensor(y_pred, message='y_pred = ')
            absPred = y_pred[:,:1]
            absTrue = y_true[:,:1]
            #absPred = print_tensor(absPred, message='absPred = ')
            #absTrue = print_tensor(absTrue, message='absTrue = ')
            #prvPred = y_pred[1:]
            prvTrue = y_true[:,1:]
            #prvTrue = print_tensor(prvTrue, message='prvTrue = ')
            anglePred = tanh(subtract(absPred, prvTrue))
            angleTrue = tanh(subtract(absTrue, prvTrue))
            grd = gradientMass*(mse(anglePred,angleTrue))
            abs = absoluteMass*(mse(absPred,absTrue))
            #return mse(anglePred,angleTrue)
            return grd+abs
        return loss
# ---------------------------------------------------------------------------------------------------------------