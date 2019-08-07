import sys
sys.path.append('01_preperation')
sys.path.append('02_lstm_keras')
sys.path.append('04_evaluation')

import datetime

from config import *
from lstmModularForexClassification import *
from sendSlack import *
import traceback

startParams = {
    'selectedSymbols'    : [["BAERCHCHF_MA6",  "BAERCHCHF_BB6",  "TIFUSUSD_MA6",  "CONDEEUR_MA6"]],
    'forexScale'         : ['1h'],            # 1s 1min 1h
    'predictionLength'   : [12],               # Number of time steps looking into the future
    'numCategories'      : [5],               # Number of categories in prediction
    'blockSize'          : [3*24],            # Number of samples in a batch/sequence
    'numNodes'           : [                  # Number of nodes in each layer of the LSTM stack [input, hidden, output]
        [10],
        [100],
        [100, 100],
        [100, 100, 100, 100],
        [1000, 1000]
    ],
    'dropout'            : [0.15],            # value of dropout, 0 means no dropout
    'numberOfEpochs'     : [1000],            # number of epochs to train
    'useMovingAvarage'   : [True],
    'debug'              : [True],
    'reloadData'         : [False],
}

# --- Config ----------------------------------------------------------------------------------------------------
MODEL_NAME = 'Modular-LSTM-Forex-Test'
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
                #SendSlack.sendText('--- MODEL ERROR -----------------------------------\r\n' + traceback.format_exc())
    except Exception as error:
        traceback.print_exc()
        #SendSlack.sendText('--- TESTUNIT ERROR -----------------------------------\r\n' + traceback.format_exc())

runConfig(startParams, Config())