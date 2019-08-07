import sys
sys.path.append('01_preperation')
sys.path.append('02_lstm_keras')
sys.path.append('04_evaluation')

import datetime

from config import *
from lstmModularDuka import *
from sendSlack import *
from keras import optimizers
import traceback

startParams = {
    # examples: 'NOVNCHCHF', 'ABBNCHCHF', 'CHEIDXCHF', 'NOVNCHCHF_MA8'
    'selectedSymbols'    : [
        ["EURCHF", "EURCHF_MA6", "EURCAD_MA6", "DISUSUSD_MA6", "VIABUSUSD_MA6"]
    ], # first value will be prediction
    'absoluteMass'       : [0], # gradientMass + absoluteMass = 1.0 ! 
    'scale'              : ['10min'],       # 1s 1min 1h
    'startdate'          : ['2018-05-01'],
        'adamLR'             : [0.0005, 0.0001],       # adam learning rate
        'adamBeta_1'         : [0.9], 
        'adamBeta_2'         : [0.999], 
        'adamEpsilon'        : [None], 
        'adamDecay'          : [0.01], 
        'adamamSgrad'        : [False],
    'activation'         : [None],     # tanh, sigmoid, relu, None
    'trainsetSize'       : [0.8],
    'predictionLength'   : [6],             # Number of time steps looking into the future
    'blockSize'          : [24*6],          # Number of samples in a batch/sequence
    'numNodes'           : [                # Number of nodes in each layer of the LSTM stack [input, hidden, output]
        [500, 200],
        [1000, 500],
    ],
    'dropout'            : [0.1, 0.2, 0.05],          # value of dropout, 0 means no dropout
    'numberOfEpochs'     : [100],           # number of epochs to train
    'useGPU'             : [True],
    'debug'              : [True],
    'reloadData'         : [False],
    'name'               : ['Duka_AbsolutePred']
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