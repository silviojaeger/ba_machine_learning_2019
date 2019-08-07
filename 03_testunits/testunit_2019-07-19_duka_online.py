import sys
sys.path.append('01_preperation')
sys.path.append('02_lstm_keras')
sys.path.append('04_evaluation')

import datetime
import os
import shutil

from config import *
from lstmModularDukaOnline import *
from sendSlack import *
from keras import optimizers
import traceback

startParams = {
    # examples: 'NOVNCHCHF', 'ABBNCHCHF', 'CHEIDXCHF', 'NOVNCHCHF_SMA8'
    'selectedSymbols'    : [
        #["SIEDEEUR_SMA8", "SIEDEEUR-Volume"], 
        #["SIEDEEUR_SMA8", "SIEDEEUR-Volume", "ESPIDXEUR_SMA8"],
        ["SIEDEEUR", "SIEDEEUR-Volume", "SIEDEEUR_MA20", "SIEDEEUR_MA200", "ESPIDXEUR", "EUSIDXEUR", "DEUIDXEUR",  "HEIDEEUR"],
        #["SIEDEEUR_EMA8", "ESPIDXEUR_EMA8", "EUSIDXEUR_EMA8", "DEUIDXEUR_EMA8",  "HEIDEEUR_EMA8"],
        #['BAERCHCHF', 'CHEIDXCHF', 'ABBNCHCHF', 'ADENCHCHF', 'CLNCHCHF', 'CSGNCHCHF', 'GIVNCHCHF', 'KNINCHCHF', 'LHNCHCHF', 'LONNCHCHF', 'NESNCHCHF', 'NOVNCHCHF', 'ROGCHCHF', 'SCMNCHCHF', 'SGSNCHCHF', 'SIKCHCHF', 'SLHNCHCHF', 'SRENCHCHF', 'UBSGCHCHF', 'UHRCHCHF', 'ZURNCHCHF'],
        #["EURCHF", "EURCHF_EMA20", "EURCHF_EMA80", "EURCHF-Volume"]
    ], # first value will be prediction
    'absoluteMass'       : [0], # gradientMass + absoluteMass = 1.0 ! 
    'startdate'          : ['2000-01-01'],
    'scale'              : ['15min', '1h', '4h'], # 1s 1min 1h
        'adamLR'             : [0.0001], # adam learning rate
        'adamBeta_1'         : [0.9], 
        'adamBeta_2'         : [0.999], 
        'adamEpsilon'        : [None], 
        'adamDecay'          : [0.0], 
        'adamAmsgrad'        : [True],
    'activation'         : ['relu'], # tanh, sigmoid, relu, None
    'trainsetSize'       : [0.2],
    'predictionLength'   : [1], # Number of time steps looking into the future
    'blockSize'          : [400], # Number of datapoints in a sample/block
    'stackSize'          : [1],
    'numNodes'           : [ # Number of nodes in each layer of the LSTM stack [input, hidden, output]
        [1200],
        [400, 200, 10]
    ], 
    'dropout'            : [0.25], # value of dropout, 0 means no dropout
    'numberOfEpochs'     : [3], # number of epochs to train
    'batchSize'          : [None], # number of samples till the weights get updates, None means default value, -1 means one batch per epoch
    'debug'              : [False],
    'showPlots'          : [False], # debug required
    'reloadData'         : [False], 
    'useGPU'             : [True],
    'lstmActivationCPU'  : ['relu'], # useGPU required
    'name'               : ['Online'], 
    'specialTimeWindows' : [[]], # additional past time data in blocksizes. e.g. [[12]] adds a data point from 12*blocksize timeunits ago
    'lever'              : [1000],
    'balance'            : [0],
    'slowDownChart'      : [0] # in seconds
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
