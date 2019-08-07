import json
import os
import sys
import pandas as pd
import urllib.request
import requests
import _pickle as pickle
import shutil
import traceback
from company import *
from logger import *
from currency import *

# --- CollectorConfig -------------------------------------------------------------------------------------------
class CollectorForexConfig:
    workdir = os.path.join(os.getcwd(), "data_temp", "collectorForex")
# ---------------------------------------------------------------------------------------------------------------


# --- Collector -------------------------------------------------------------------------------------------------
class CollectorForex:
    """ collects data from different sources and stores data to a pickle for fast access
        usage: collector = Collector(replace=False)

        ! some stocks may be skipt due to data lacks in WTD database !
    """
    def __init__(self, replace=False, forexScale="1h", symbols=[]): 
        self.forexScale = forexScale
        self.symbols = symbols
        self.replace = replace
        self.__collect()

    def getBySymbol(self, symbol):
        """ get a company by symbol eg. ABBN.SW
            usage: collector.getBySymbol(symbol)
        """
        for currency in self.currencies:
            if symbol == currency.symbol: return currency       

    def __collect(self):
        # prepare
        if not os.path.exists(CollectorForexConfig.workdir): os.makedirs(CollectorForexConfig.workdir)
        self.currencies = [ Currency(c, None) for c in self.symbols ]
        
        # load data
        failedCurrencies = []
        lastError = ""
        for currency in self.currencies:
            try:
                fileCurrency = os.path.join(CollectorForexConfig.workdir, f"{currency.symbol}.pickle")
                # check for forex data
                if not os.path.isfile(fileCurrency) or self.replace:
                    # download and parse
                    self.__getDf(currency) # get forex history in ticks
                    with open(fileCurrency, 'wb') as file: pickle.dump(self.currencies, file)
                else:
                    self.currencies = pickle.load(open(fileCurrency, 'rb'))
            except: 
                failedCurrencies += [currency.symbol]
                lastError = traceback.format_exc()
        sys.stdout.write('\n')
        # clean up
        if len(failedCurrencies) > 0: print(f'failed to download: {failedCurrencies}\r\n{lastError}')

    def __getDf(self, currency):
        directory = os.path.join(CollectorForexConfig.workdir, currency.symbol)
        if not os.path.exists(directory): 
            os.makedirs(directory)
            raise (f'No csv files found in {directory}')
        files = os.listdir(directory)
        dfMonths = []
        progress = 0
        for csvFile in files:
            progress += 1
            sys.stdout.write(f'\r{currency.symbol} {Logger.bulk(progress, len(files))} {csvFile}    ')
            dfMonth = pd.read_csv(os.path.join(directory, csvFile), delimiter=",", encoding='latin1', parse_dates=True, header=None, index_col=0)
            dfMonth.columns = ['Ask','Bid','AskVolume','BidVolume']
            dfMonth = dfMonth.resample(self.forexScale).mean()
            dfMonth = dfMonth[dfMonth.index.weekday<5]
            dfMonths += [dfMonth]
        currency.df = pd.concat(dfMonths)
        return currency
# ---------------------------------------------------------------------------------------------------------------