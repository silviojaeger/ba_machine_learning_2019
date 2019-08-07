from duka.app import app
from duka.core import valid_date, set_up_signals
from duka.core.utils import valid_timeframe, TimeFrame
from pandas import DataFrame
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta

# --- Config ----------------------------------------------------------------------------------------------------
class Config():
    END_DATE = date(2019, 1, 1)
    MONTHS = 1
    PATH = '/'
    CANDLE = TimeFrame.TICK
    THREADS = 1
    CSV_HEADER = False
    FOREX = ['XAUUSD']
# ---------------------------------------------------------------------------------------------------------------

def grabTimerange():
    start = Config.END_DATE - relativedelta(months=Config.MONTHS)
    for month in range(Config.MONTHS):
        end = start + relativedelta(months=1)
        print('\r\nRequest from ', str(start), ' to ', str(end - relativedelta(days=1)))
        set_up_signals()
        app(
            Config.FOREX, 
            start,
            end, 
            Config.THREADS,  
            Config.CANDLE, 
            Config.PATH,
            Config.CSV_HEADER) 
        start = end

# --- Start -----------------------------------------------------------------------------------------------------
def main():
    try:
        grabTimerange()
    except:
        print('unknown error' + "\r\n\r\n" + str(traceback.format_exc()))
#
if __name__ == "__main__": main()
# ---------------------------------------------------------------------------------------------------------------