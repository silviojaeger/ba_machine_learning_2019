import sys
sys.path.append('01_preperation')
sys.path.append('02_lstm_keras')
sys.path.append('04_evaluation')

import datetime

from config import *
from lstmModular import *
from sendSlack import *
import traceback

startParams = {
    'selectedSymbol'     : ['NOVN.SW', 'ROG.SW'],
    'currencySource'     : ['CHF', None],
    'currenciesToCompare': [['EUR', 'USD']],
    'correlatingSymbols' : [6, 1],
    'correlationLag'     : [60],
    'correlationTime'    : [1080],
    'correlationMethod'  : ['spearman'],
    'predictionSize'     : [5],               # Number of time steps looking into the future
    'blockSize'          : [120],             # Number of samples in a batch/sequence
    'numNodes'           : [                  # Number of nodes in each layer of the LSTM stack [input, hidden, output]
        [120, 480, 480, 480, 480, 480, 480, 480],
        [120, 2400, 2400, 120],
        [120, 120, 120]                
    ],
    'dropout'            : [0.09, 0.18],      # value of dropout, 0 means no dropout
    'numberOfEpochs'     : [2000],            # number of epochs to train
    'modelTrained'       : [False],
    'shift'              : [False, True]
}

# --- Config ----------------------------------------------------------------------------------------------------
MODEL_NAME = 'Modular-LSTM-Test'
time_stamp = str(datetime.datetime.utcnow()).replace(":", "-")  # date-time to name folders and data
workdir = os.path.join(os.getcwd(), "logs", f'{MODEL_NAME}_{time_stamp}')
# ---------------------------------------------------------------------------------------------------------------

def runConfig(params, conf):
    try:
        if len(params) > 0:
            # continue interation
            element = list(params.keys())[0]
            values = params.pop(element)
            for value in values:
                conf[element] = value
                runConfig(params.copy(), conf)
        else:
            conf.print()
            try:
                lstmModular(conf, workdir)
            except Exception as error:
                traceback.print_exc()
                SendSlack.sendText('--- MODEL ERROR -----------------------------------\r\n' + traceback.format_exc())
    except Exception as error:
        traceback.print_exc()
        SendSlack.sendText('--- TESTUNIT ERROR -----------------------------------\r\n' + traceback.format_exc())

runConfig(startParams, Config())