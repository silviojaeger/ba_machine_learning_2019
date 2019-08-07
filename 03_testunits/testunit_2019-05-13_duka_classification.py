import sys
sys.path.append('01_preperation')
sys.path.append('02_lstm_keras')
sys.path.append('04_evaluation')

import datetime

from config import *
from lstmModularDukaClassification import *
from sendSlack import *
from keras import optimizers
import traceback

startParams = {
    # examples: 'NOVNCHCHF', 'ABBNCHCHF', 'CHEIDXCHF', 'NOVNCHCHF_MA8', -ln(0.2)=1.60944
    # ["BRENTCMDUSD_MA50", "APAUSUSD_MA50", "APCUSUSD_MA50"]
    'selectedSymbols'    : [
      ["SIEDEEUR_SMA8", "SIEDEEUR-Volume", "ESPIDXEUR_SMA8", "EUSIDXEUR_SMA8", "DEUIDXEUR_SMA8", "HEIDEEUR_SMA8"]
    ], # first value will be prediction
    'startdate'          : ['2018-05-01'],
    'scale'              : ['15min'],       # 1s 1min 1h
        'adamLR'             : [0.001],     # adam learning rate, standard 0.01
        'adamBeta_1'         : [0.9], 
        'adamBeta_2'         : [0.999], 
        'adamEpsilon'        : [None],      # None: will automatically set to default value 
        'adamDecay'          : [0.0], 
        'amsgrad'            : [True],
    'activation'         : ['relu'],        # tanh, sigmoid, relu (last dense layer uses SOFTMAX)
    'trainsetSize'       : [0.8],
    'predictionLength'   : [12,24],         # Number of time steps looking into the future
    'numCategories'      : [3],             # Number of categories in prediction
    'blockSize'          : [2*24*5],            # Number of samples in a sequence as input
    'numNodes'           : [                # Number of nodes in each layer of the LSTM stack [input, hidden, output]
        [1200]
    ],
    'dropout'            : [0.2, 0.4],      # value of dropout, 0 means no dropout
    'numberOfEpochs'     : [50],           # number of epochs to train
    'batchSize'          : [4*24*5],        # number of samples till the weights get updates, None means default value, -1 means one batch per epoch
    'debug'              : [False],
    'name'               : ['Classification_DokuGridsearch'],
    'useGPU'             : [True],
    'checkOverValue'     : [0.4],           #percentegage of rigth direction over this value
    'shuffleInput'       : [False]
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