import json
import os
import sys
import pandas as pd
import urllib.request
import requests
import _pickle as pickle
import shutil
from company import *
from logger import *
from currency import *

# --- CollectorConfig -------------------------------------------------------------------------------------------
class CollectorConfig:
    workdir        = os.path.join(os.getcwd(), "data_temp", "collector")
    workdirForex   = os.path.join(workdir, "forex", 'daily')
    workdirForTick = os.path.join(os.getcwd(), "data_temp", "forex", 'tickly')
    workdirSix     = os.path.join(workdir, "six", 'daily')
    prices         = "close" # open close high low volume
    sixStocks      = "https://www.six-group.com/exchanges/shares/companies/download/issuers_all_en.csv"
    #wtdToken      = 'NOCj3No1I1qlGzHoVG7qLlFxOgWLi2jK1MNgbcZc7RBf8enQ1DbJ5rbLKIdT' # SJ
    #wtdToken      = 'npeK0CbwL7wj5hs9Z6RYxXkKX1T7VDlJHJDLwsk3O23btbFBKQz5v1AteGVI' # SJ
    #wtdToken      = 'pJvWczZNwt6mVZHOLDM8qrN9HH7jI5n0xJMX2SQO3mc04P0xXiOvbNDnaBZZ' # random
    #wtdToken      = 'QX5X7EfPTb3n2yxzuPneaslfHtKJTpPnOp7CB9kbFwIE8O3jhCU5vm59Znjg' # JE
    wtdToken       = 'qUjmLPwIAuS4pzrDZbPf26r6HwkFSQU2QbMxdBvVeYevNuvUZ2DFTtCuujVA' # ba.aktienkurse@gmail.com
    wtdUrl         = 'https://www.worldtradingdata.com/api/v1/history'
    wtdForexUrl    = 'https://www.worldtradingdata.com/api/v1/forex_history'
# ---------------------------------------------------------------------------------------------------------------


# --- Collector -------------------------------------------------------------------------------------------------
class Collector:
    """ collects data from different sources and stores data to a pickle for fast access
        usage: collector = Collector(replace=False)

        ! some stocks may be skipt due to data lacks in WTD database !
    """
    def __init__(self, replace=False): 
        # load data
        fileCompanies        = os.path.join(CollectorConfig.workdir, "companies.pickle")
        fileCurrencies       = os.path.join(CollectorConfig.workdir, "currencies.pickle")
        # check for wtd data
        if not os.path.isfile(fileCompanies) or not os.path.isfile(fileCurrencies) or replace: 
            shutil.rmtree(CollectorConfig.workdir, ignore_errors=True)
            self.__collectForex()
            self.__collectSpi()
            with open(fileCompanies       , 'wb') as file: pickle.dump(self.companies       , file)
            with open(fileCurrencies      , 'wb') as file: pickle.dump(self.currencies      , file)
        else: 
            self.companies        = pickle.load(open(fileCompanies , 'rb'))
            self.currencies       = pickle.load(open(fileCurrencies, 'rb'))

    def getBySymbol(self, symbol):
        """ get a company by symbol eg. ABBN.SW
            usage: collector.getBySymbol(symbol)
        """
        for company in self.companies:
            if symbol == company.symbol: return company
        for currency in self.currencies:
            if symbol == currency.symbol: return currency   
        for currency in self.currenciesTickly:
            if symbol == currency.symbol: return currency       

    def __collectForex(self):
        # prepare
        if not os.path.exists(CollectorConfig.workdirForex): os.makedirs(CollectorConfig.workdirForex)
        self.currencies = [
            Currency('CHF', ['USD', 'EUR', 'GBP', 'CNY'])
        ]
        # download and parse
        self.__getDfForex() # get forex history

    def __getDfForex(self):
        # download json from wtd
        print(f'download forex symbols from WTD: {[c.symbol for c in self.currencies]}')
        for currency in self.currencies:
            progress = 0
            for compareCurr in currency.relations:
                progress += 1
                path = os.path.join(CollectorConfig.workdirForex, 'forex.'+currency.symbol+compareCurr+'.json')
                sys.stdout.write(f'\r{currency.symbol} {Logger.bulk(progress, len(currency.relations))} {compareCurr}    ')
                params = dict(
                    base=currency.symbol,
                    convert_to=compareCurr,
                    sort='oldest',
                    api_token=CollectorConfig.wtdToken
                )
                data = requests.get(url=CollectorConfig.wtdForexUrl, params=params).json()
                if 'Message' in data: raise ValueError()
                with open(path, 'w') as outfile:
                    json.dump(data, outfile)
        sys.stdout.write('\r\n')
        # convert wtd json to df
        print(f'load forex symbols from json: {[c.symbol for c in self.currencies]}')
        for currency in self.currencies:
            progress = 0
            for compareCurr in currency.relations:
                progress += 1
                path = os.path.join(CollectorConfig.workdirForex, 'forex.'+currency.symbol+compareCurr+'.json')
                sys.stdout.write(f'\r{currency.symbol} {Logger.bulk(progress, len(currency.relations))} {compareCurr}    ')
                with open(path, 'r') as file:
                    json1 = json.load(file)
                    data = [[]]
                    for r in json1['history']:
                        data += [ pd.Timestamp(r), json1['history'][r] ]
                currency.df[compareCurr] = pd.Series(
                    index = [pd.Timestamp(date) for date in json1['history']],
                    data =  [float(json1['history'][date]) for date in json1['history']]
                )
        sys.stdout.write('\n')

    def __collectSpi(self):
        # prepare 
        if not os.path.exists(CollectorConfig.workdirSix): os.makedirs(CollectorConfig.workdirSix)
        if not os.path.isfile(os.path.join(CollectorConfig.workdirSix, "six_stocks.csv")):
            urllib.request.urlretrieve(CollectorConfig.sixStocks, os.path.join(CollectorConfig.workdirSix, "six_stocks.csv"))
        # create dataframe with company infos
        df = pd.read_csv(os.path.join(CollectorConfig.workdirSix, "six_stocks.csv"), delimiter=";", encoding='latin1')
        # get all spi companies
        df = df[df.iloc[:,6].str.contains('SPI', na=False)].reset_index(drop=True)
        self.companies = [Company(row[0], row[1] + ".SW", row[3]) for index, row in df.iterrows()]
        # download and parse
        self.__getDfSpi() # get stock history

    def __getDfSpi(self):
        # download json from wtd
        completedCompanies = []
        failedCompanies = []
        print('download symbols from WTD')
        progress = 0
        for company in self.companies:
            progress += 1
            try:
                symbol = company.symbol
                path = os.path.join(CollectorConfig.workdir, "six", 'daily', 'stock.'+symbol+'.json')
                # download
                sys.stdout.write(f'\r{Logger.bulk(progress, len(self.companies))} {company.symbol}    ')
                params = dict(
                    symbol=symbol,
                    sort='oldest',
                    api_token=CollectorConfig.wtdToken
                )
                data = requests.get(url=CollectorConfig.wtdUrl, params=params).json()
                if 'Message' in data: raise ValueError()
                with open(path, 'w') as outfile:
                    json.dump(data, outfile)
                completedCompanies.append(company)
            except: 
                failedCompanies += [company.symbol]
        sys.stdout.write('\n')
        # clean up
        if len(failedCompanies) > 0: print(f'failed to download: {failedCompanies}')
        self.companies = completedCompanies
        # convert wtd json to df
        completedCompanies = []
        failedCompanies = []
        print('load symbols from json')
        progress = 0
        for company in self.companies:
            progress += 1
            try:
                symbol = company.symbol
                path = os.path.join(CollectorConfig.workdir, "six", 'daily', 'stock.'+symbol+'.json')
                sys.stdout.write(f'\r{Logger.bulk(progress, len(self.companies))} {company.symbol}    ')
                with open(path, 'r') as file:
                    json1 = json.load(file)
                    data = [[]]
                    for r in json1['history']:
                        data += [ pd.Timestamp(r), json1['history'][r]['open'] ]
                company.df = pd.DataFrame(
                    index=[pd.Timestamp(date) for date in json1['history']],
                    data={ 
                        'open'   : [float(json1['history'][date]['open'   ]) for date in json1['history']],
                        'close'  : [float(json1['history'][date]['close'  ]) for date in json1['history']],
                        'high'   : [float(json1['history'][date]['high'   ]) for date in json1['history']],
                        'low'    : [float(json1['history'][date]['low'    ]) for date in json1['history']],
                        'volume' : [float(json1['history'][date]['volume' ]) for date in json1['history']],
                    }
                )
                completedCompanies.append(company)
            except: 
                failedCompanies += [company.symbol]
        sys.stdout.write('\n')
        # clean up
        if len(failedCompanies) > 0: print(f'failed to parse {failedCompanies}')
        self.companies = completedCompanies
# ---------------------------------------------------------------------------------------------------------------