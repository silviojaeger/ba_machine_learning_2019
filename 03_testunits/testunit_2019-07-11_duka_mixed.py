import sys
sys.path.append('01_preperation')
sys.path.append('02_LSTM_Keras')
sys.path.append('04_evaluation')

import datetime
import os
import shutil

from config import *
from lstmModularDukaMixed import *
from sendSlack import *
from keras import optimizers
import traceback

startParams = {
    # examples: 'NOVNCHCHF', 'ABBNCHCHF', 'CHEIDXCHF', 'NOVNCHCHF_SMA8'
    'selectedSymbols'    : [
        #["SIEDEEUR_SMA8", "SIEDEEUR-Volume"], 
        #["SIEDEEUR_SMA8", "SIEDEEUR-Volume", "ESPIDXEUR_SMA8"],
        ["SIEDEEUR_EMA8", "SIEDEEUR-Volume", "ESPIDXEUR_EMA8", "EUSIDXEUR_EMA8", "DEUIDXEUR_EMA8",  "HEIDEEUR_EMA8"]
        #['BAERCHCHF_EMA8', 'BAERCHCHF_EMA8', 'BAERCHCHF_EMA8', 'CHEIDXCHF_EMA8', 'ABBNCHCHF_EMA8', 'ADENCHCHF_EMA8', 'CLNCHCHF_EMA8', 'CSGNCHCHF_EMA8', 'GIVNCHCHF_EMA8', 'KNINCHCHF_EMA8', 'LHNCHCHF_EMA8', 'LONNCHCHF_EMA8', 'NESNCHCHF_EMA8', 'NOVNCHCHF_EMA8', 'ROGCHCHF_EMA8', 'SCMNCHCHF_EMA8', 'SGSNCHCHF_EMA8', 'SIKCHCHF_EMA8', 'SLHNCHCHF_EMA8', 'SRENCHCHF_EMA8', 'UBSGCHCHF_EMA8', 'UHRCHCHF_EMA8', 'ZURNCHCHF_EMA8'],
        #["EURUSD", "EURUSD_EMA20", "EURUSD_EMA80", "EURUSD-Volume"]
    ], # first value will be prediction
    'absoluteMass'       : [1], # gradientMass + absoluteMass = 1.0 ! 
    'startdate'          : ['2018-01-01'],
    'scale'              : ['1min','5min','15min'], # 1s 1min 1h
        'adamLR'             : [0.001], # adam learning rate
        'adamBeta_1'         : [0.9], 
        'adamBeta_2'         : [0.999], 
        'adamEpsilon'        : [None], 
        'adamDecay'          : [0.0], 
        'adamAmsgrad'        : [True],
    'activation'         : ['relu'], # tanh, sigmoid, relu, None
    'trainsetSize'       : [0.8],
    'predictionLength'   : [1], # Number of time steps looking into the future
    'blockSize'          : [100], # Number of datapoints in a sample/block
    'numNodes'           : [ # Number of nodes in each layer of the LSTM stack [input, hidden, output]
        [1200]
    ],
    'dropout'            : [0.15], # value of dropout, 0 means no dropout
    'numberOfEpochs'     : [400], # number of epochs to train
    'batchSize'          : [500], # number of samples till the weights get updates, None means default value, -1 means one batch per epoch
    'debug'              : [False],
    'showPlots'          : [False], # debug required
    'reloadData'         : [False], 
    'useGPU'             : [True], 
    'shuffle'            : [False],
    'lstmActivationCPU'  : ['relu'], # useGPU required
    'name'               : ['EURCHF'], 
    'specialTimeWindows' : [[2,3,4,5]] # additional past time data in blocksizes. e.g. 12 adds a data point from [12*blocksize] timeunits ago
}

# --- Config ----------------------------------------------------------------------------------------------------
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

if __name__ == '__main__':
    runConfig(startParams, Config())
