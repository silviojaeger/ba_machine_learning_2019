import sys
sys.path.append('01_preperation')
sys.path.append('02_lstm_keras')
sys.path.append('04_evaluation')

import datetime

from config import *
from lstmModularDuka import *
from sendSlack import *
import traceback

startParams = {
    # examples: 'NOVNCHCHF', 'ABBNCHCHF', 'CHEIDXCHF', 'NOVNCHCHF_MA8'
    'selectedSymbols'    : [
        ['SINUS']
    ], # first value will be prediction
    'scale'              : ['10min'],       # 1s 1min 1h
    'optimizer'          : ['adam'], # adadelta
    'activation'         : ['softmax'], # tanh, sigmoid, relu
    'trainsetSize'       : [0.8],
    'predictionLength'   : [2],             # Number of time steps looking into the future
    'numCategories'      : [5],             # Number of categories in prediction
    'blockSize'          : [20],            # Number of samples in a batch/sequence
    'numNodes'           : [                # Number of nodes in each layer of the LSTM stack [input, hidden, output]
        [10]
    ],
    'dropout'            : [0],            # value of dropout, 0 means no dropout
    'numberOfEpochs'     : [1],           # number of epochs to train
    'debug'              : [False],
    'reloadData'         : [False],
    'name'               : ['sinus']
}

# --- Config ----------------------------------------------------------------------------------------------------
MODEL_NAME = 'Dukascopy-LSTM-Sinus-Test'
time_stamp = str(datetime.datetime.utcnow()).replace(":", "-")  # date-time to name folders and data
workdir = os.path.join(os.getcwd(), "logs", f'{startParams["name"][0]}_{time_stamp}')
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
                lstmModularDuka(conf, workdir)
            except Exception as error:
                traceback.print_exc()
                if not conf['debug']: SendSlack.sendText('--- MODEL ERROR -----------------------------------\r\n' + traceback.format_exc())
    except Exception as error:
        traceback.print_exc()
        if not conf['debug']: SendSlack.sendText('--- TESTUNIT ERROR -----------------------------------\r\n' + traceback.format_exc())

runConfig(startParams, Config())