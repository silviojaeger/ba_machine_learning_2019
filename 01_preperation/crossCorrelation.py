import sys
sys.path.append('01_preperation')
sys.path.append('04_evaluation')

import pandas as pd
import tensorflow
import numpy as np
import datetime
import os
import functools
import glob
import multiprocessing
import matplotlib.pyplot as plt

from collectorDuka import *
from logger import *
from calculator import *
from slicer import *

cTime = 180*24*6
cLag = 2*24
folder = "stocks_de"

def singleCC(df, stock):
    global cTime, cLag
    ccdf = Calculator.crossCorrelationDf(df, stock, lag=cLag, time=cTime, showProgress=False)
    # index table
    srIndex = ccdf['index']
    srIndex.name = stock
    # shift table
    srShift = ccdf['shift']
    srShift.name = stock
    return (srIndex, srShift)

def crossCorrelation():
    global cTime, cLag, folder
    timespan = cTime + 2*cLag + 1 
    df = CollectorDuka(scale='10min', folder=folder, csvColumn="Close").df
    df = df.loc['2018-05-01':'2019-05-01'] # trim end, change later to trimTail
    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df.rolling(window=8).mean()
    df.drop(df.index[:-timespan], inplace=True)
    df = Calculator.scale(df)
    #df.plot()
    #plt.show()
    ccdfsIndex = []
    ccdfsShift = []
    # make cross correlation
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
    ccResult = pool.imap_unordered(functools.partial(singleCC, df), df.columns)
    pool.close() # No more work
    while (True):
        completed = ccResult._index
        sys.stdout.write(f'\r{Logger.bulk(completed, len(df.columns))} Cross Correlation    ')
        if (completed == len(df.columns)): break
        time.sleep(0.1)
    sys.stdout.write('\n')
    # get result
    ccComb  = {}
    for srIndex, srShift in ccResult:
        ccdf = pd.concat([srIndex, srShift], axis=1)
        ccdf.columns = ['index', 'shift']
        ccdf = ccdf.sort_values(ascending=False, by=['index', 'shift'])
        ccComb[srIndex.name] = ccdf
    # plot individual information 
    ccDir = os.path.join(os.getcwd(), 'logs', 'cross_correlation', folder)  
    csvDir = os.path.join(ccDir, 'csv')
    xlsDir = os.path.join(ccDir, 'xls')
    if not os.path.exists(csvDir): os.makedirs(csvDir)
    if not os.path.exists(xlsDir): os.makedirs(xlsDir)
    for sym, ccdf in ccComb.items():
        #ccdf = ccdf[ccdf['shift']>0]
        ccdf = ccdf.sort_values(ascending=False, by=['index', 'shift'])
        ccdf.to_csv(  os.path.join(csvDir, f'{sym}.csv'))
        ccdf.to_excel(os.path.join(xlsDir, f'{sym}.xls'))

if __name__ == '__main__':
    crossCorrelation()