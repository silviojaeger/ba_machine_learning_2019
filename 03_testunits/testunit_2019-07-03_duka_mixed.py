import sys
sys.path.append('01_preperation')
sys.path.append('02_lstm_keras')
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
        ["SIEDEEUR_SMA8", "SIEDEEUR-Volume", "ESPIDXEUR_SMA8", "EUSIDXEUR_SMA8", "DEUIDXEUR_SMA8",  "HEIDEEUR_SMA8"],
        #['CHEIDXCHF_SMA8', 'CHEIDXCHF-Volume','ABBNCHCHF_SMA8', 'ADENCHCHF_SMA8', 'BAERCHCHF_SMA8', 'CLNCHCHF_SMA8', 'CSGNCHCHF_SMA8', 'GIVNCHCHF_SMA8', 'KNINCHCHF_SMA8', 'LHNCHCHF_SMA8', 'LONNCHCHF_SMA8', 'NESNCHCHF_SMA8', 'NOVNCHCHF_SMA8', 'ROGCHCHF_SMA8', 'SCMNCHCHF_SMA8', 'SGSNCHCHF_SMA8', 'SIKCHCHF_SMA8', 'SLHNCHCHF_SMA8', 'SRENCHCHF_SMA8', 'UBSGCHCHF_SMA8', 'UHRCHCHF_SMA8', 'ZURNCHCHF_SMA8'],
        #["EURUSD_SMA20", "EURUSD", "EURUSD-Volume", "EURUSD_MA80"]
        #["CUSUSD_EMA2", "CLNCHCHF_EMA2", "FRAIDXEUR_EMA2", "DEUIDXEUR_EMA2", "CSGNCHCHF_EMA2", "UBSGCHCHF_EMA2"]
    ], # first value will be prediction
    'absoluteMass'       : [0], # gradientMass + absoluteMass = 1.0 ! 
    'startdate'          : ['2019-02-01'],
    'scale'              : ['15min'], # 1s 1min 1h
        'adamLR'             : [0.001], # adam learning rate
        'adamBeta_1'         : [0.9], 
        'adamBeta_2'         : [0.999], 
        'adamEpsilon'        : [None], 
        'adamDecay'          : [0.0], 
        'adamAmsgrad'        : [True],
    'activation'         : ['relu'], # tanh, sigmoid, relu, None
    'trainsetSize'       : [0.8],
    'predictionLength'   : [12], # Number of time steps looking into the future
    'blockSize'          : [2*4*24], # Number of samples in a batch/sequence
    'numNodes'           : [ # Number of nodes in each layer of the LSTM stack [input, hidden, output]
        [1200]
    ],
    'dropout'            : [0.4], # value of dropout, 0 means no dropout
    'numberOfEpochs'     : [1200], # number of epochs to train
    'batchSize'          : [4*24*5], # number of samples till the weights get updates, None means default value, -1 means one batch per epoch
    'debug'              : [False],
    'showPlots'          : [False], # debug required
    'reloadData'         : [False],
    'useGPU'             : [True], 
    'shuffle'            : [False],
    'lstmActivationCPU'  : ['relu'], # useGPU required
    'name'               : ['GlobalGridSearch'], 
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