import sys
sys.path.append('01_preperation')
sys.path.append('02_lstm_keras')

import datetime

from config import *
from lstmModular import * 

startParams = {
    'selectedSymbol'     : ['ABBN.SW', 'NESN.SW', 'NOVN.SW'],
    'correlatingSymbols' : [4],
    'correlationLag'     : [30],
    'correlationTime'    : [90],
    'correlationMethod'  : ['spearman'],
    'predictionSize'     : [1],               # Number of time steps looking into the future
    'blockSize'          : [60],              # Number of samples in a batch/sequence
    'numNodes'           : [                  # Number of nodes in each layer of the LSTM stack [input, hidden, output]
        [60, 120, 120, 120, 60],               
        [60, 60]
    ],
    'dropout'            : [0.1, 0.01],       # value of dropout, 0 means no dropout
    'numberOfEpochs'     : [800],             # number of epochs to train
    'modelTrained'       : [False],
    'shift'              : [False]
}

# --- Config ----------------------------------------------------------------------------------------------------
MODEL_NAME = 'Modular-LSTM-Test'
time_stamp = str(datetime.datetime.utcnow()).replace(":", "-")  # date-time to name folders and data
workdir = os.path.join(os.getcwd(), "logs", f'{MODEL_NAME}_{time_stamp}')
# ---------------------------------------------------------------------------------------------------------------

def runConfig(params, conf):
    if len(params) > 0:
        # continue interation
        element = list(params.keys())[0]
        values = params.pop(element)
        for value in values:
            conf[element] = value
            runConfig(params.copy(), conf)
    else:
        conf.print()
        lstmModular(conf, workdir)
        
runConfig(startParams, Config())