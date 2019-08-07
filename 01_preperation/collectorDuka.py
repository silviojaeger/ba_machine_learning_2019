import os
import sys
import pandas as pd
import _pickle as pickle
import traceback
import glob
import re
import multiprocessing
import time
from logger import *
from slicer import *


# --- CollectorConfig -------------------------------------------------------------------------------------------
class CollectorDukaConfig:
    def getWorkdir():
        usualDir = os.path.join(os.getcwd(), "data_temp", "collectorDuka")
        if(os.path.isdir(usualDir)): return usualDir
        else: return os.path.join("U:")
    workdir = getWorkdir()
    priceChart = 'Ask'
    plotTraceback = False
    csvDelimiter = ';'
    decimal = '.'
# ---------------------------------------------------------------------------------------------------------------


# --- Collector -------------------------------------------------------------------------------------------------
class CollectorDuka:
    """ collects data from csv files, downloaded from dukascopy.com
    """
    def __init__(self, scale="1h", symbols=[], folder="*", csvColumn='*'): 
        """ csvColumn = '*' # *, Close. Close column name is only symbol name. """
        self.scale = scale
        self.symbols = symbols
        self.folder = folder
        self.csvColumn = csvColumn
        self.__collect()

    def loadSymbol(self, sym):
        try:
            path = os.path.join(CollectorDukaConfig.workdir, self.folder, f'{sym}*{CollectorDukaConfig.priceChart}*.csv')
            csvFile = glob.glob(path)[0]
            #progress += 1
            #sys.stdout.write(f'\r{sym} {Logger.bulk(progress, len(self.symbols))} {csvFile}    ')
            dfSym = pd.read_csv(csvFile, delimiter=CollectorDukaConfig.csvDelimiter, index_col=0, parse_dates=True, decimal=CollectorDukaConfig.decimal)
            dfSym.columns = ['Open','High','Low','Close','Volume']
            dfSym = dfSym.resample(self.scale).mean()
            dfSym = dfSym[dfSym['Volume']>0]
            # get only one column 
            if self.csvColumn != "*":
                srSym = dfSym[self.csvColumn]
                srSym.name = sym
                return srSym
            else:
                dfSym.columns = [sym+'-Open',sym+'-High',sym+'-Low',sym,sym+'-Volume']
                return dfSym
        except:
            if csvFile == None: csvFile = sym
            print(f'\rfailed to load: {os.path.basename(csvFile)}             ')
            if CollectorDukaConfig.plotTraceback: traceback.print_exc()

    def __collect(self):
        # prepare
        if not os.path.exists(CollectorDukaConfig.workdir): os.makedirs(CollectorDukaConfig.workdir)
        csvFiles = []
        if self.symbols == []:
            path = os.path.join(CollectorDukaConfig.workdir, self.folder, '*_Ask_*.csv')
            self.symbols = glob.glob(path)
            self.symbols = [re.sub(r"_.*", '', os.path.basename(i), 0) for i in self.symbols]
        # load data
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        dfArray = pool.imap_unordered(self.loadSymbol, self.symbols)
        pool.close() # No more work
        while (True):
            completed = dfArray._index
            sys.stdout.write(f'\r{Logger.bulk(completed, len(self.symbols))} Loading Symbols')
            if (completed == len(self.symbols)): break
            time.sleep(0.1)
        sys.stdout.write('\n')
        # concat
        print(f'Concat Data')
        #dfArray = Slicer.trimHeadOfArray(dfArray)
        self.df = pd.concat(dfArray, axis=1, sort=False)
        self.df = Slicer.trimHead(self.df)
        self.df = Slicer.trimTail(self.df)
        columns = self.symbols.copy()
        columns.pop(0)
        self.df[columns] = self.df[columns].fillna(method="ffill").fillna(method="bfill")
        self.df = self.df.dropna()
        print(f"Data available from {self.df.index[0]} to {self.df.index[-1]}")
# ---------------------------------------------------------------------------------------------------------------
