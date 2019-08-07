import sys
sys.path.append('01_preperation')

import pandas as pd
import tensorflow
import numpy as np
import datetime
import os

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten, CuDNNLSTM
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras import optimizers
from keras import metrics

from collector import *
from calculator import *
from slicer import *
from logger import *
from config import *
from sendSlack import *


# --- Config ----------------------------------------------------------------------------------------------------
MODEL_NAME = 'LSTM-Test'
time_stamp = str(datetime.datetime.utcnow()).replace(":", "-")  # date-time to name folders and data
tensorboardLogDir = os.path.join(os.getcwd(), "logs", f'{MODEL_NAME}_{time_stamp}')
tensorboard = TensorBoard(log_dir=tensorboardLogDir, histogram_freq=0, write_graph=True, write_images=True)
# ---------------------------------------------------------------------------------------------------------------


# --- Model configuration ---------------------------------------------------------------------------------------
conf = Config({
    'selectedSymbol'     : 'ABBN.SW',
    'correlatingSymbols' : 4,
    'correlationLag'     : 30,
    'correlationTime'    : 90,
    'correlationMethod'  : 'spearman',
    'predictionSize'     : 1,    #[1, 3]                     # Number of time steps looking into the future
    'blockSize'          : 120,                              # Number of samples in a batch/sequence
    'numNodes'           : [120, 120, 60, 30],               # Number of nodes in each layer of the LSTM stack
    'useDropout'         : True,                             # choose dropout
    'dropout'            : 0.001,                            # value of dropout
    'numberOfEpochs'     : 1                                 # number of epochs to train
})
# ---------------------------------------------------------------------------------------------------------------


# --- Preperation -----------------------------------------------------------------------------------------------
print('Collect Companies')
col = Collector(replace=False)
companies = col.companies
currencies = col.currencies

for company in companies:
    company.diffSeries = company.df['close'].diff().iloc[:-1]

targetCompany = col.getBySymbol(conf['selectedSymbol'])
corrList = []
progress = 0
print(f'Correlate symbol {targetCompany.symbol}')
for company in companies:
    progress += 1
    sys.stdout.write(f'\r{Logger.bulk(progress, len(companies))} {company.symbol}    ')
    corrList += [Calculator.crossCorrelation(targetCompany.diffSeries, company.diffSeries, lag=conf['correlationLag'], time=conf['correlationTime'], method=conf['correlationMethod'])]
cross = pd.DataFrame(
        data =  corrList,
        index = [company.symbol for company in companies],
        columns = ['shift', 'index'])

cross['index'] = cross['index'].abs() # build abs
cross = cross[cross.index != conf['selectedSymbol']] # delete self correlation
cross = cross[cross['shift'] > 0] # select all negative shifts
cross = cross.sort_values('index', ascending=False) # sort desc
cross = cross[:conf['correlatingSymbols']] # get top correlations as defined in config
print(f'Correlation Result for {company.symbol}')
print(cross)

targetSymbols = cross.index
corrList = [col.getBySymbol(symbol).df['close'] for symbol in targetSymbols]
# ---------------------------------------------------------------------------------------------------------------


# --- Slice -----------------------------------------------------------------------------------------------------
print('Slice Dataframe')
close  = targetCompany.df['close']
volume = targetCompany.df['volume']
corrList = [df for df in corrList]
df = pd.concat(
    [close] + 
    [volume] +
    corrList, axis=1, sort=False)
df.columns = (
    [targetCompany.symbol] +
    [targetCompany.symbol + ".volume"] + 
    list(targetSymbols))

df = Slicer.trim(df)
df = Calculator.scale(df)

train, test    = Slicer.split(df, trainsetSize=0.8)
trainX, trainY = Slicer.slice(train, block=conf['blockSize'], prediction=conf['predictionSize'])
testX, testY   = Slicer.slice(test, block=conf['blockSize'], prediction=conf['predictionSize'])

# debug
print(f"TrainData: \r\n{train[-2:]}")
print(f"TestData: \r\n{test[-2:]}")

# ---------------------------------------------------------------------------------------------------------------


# --- create model ----------------------------------------------------------------------------------------------
model = Sequential()

# build the layers
nbrOfLayers = len(conf['numNodes'])
iterNbr = 0
if nbrOfLayers != 0: 
    for i in conf['numNodes']:
        iterNbr+=1
        print(f'Iteration: {iterNbr}')
        if(iterNbr == 1):
            # input layer
            print(f'Add LSTM input Layer with {i} Nodes')
            model.add(CuDNNLSTM(i, input_shape=(conf['blockSize'], len(df.columns)), return_sequences=True))
            if conf['useDropout']:
                model.add(Dropout(conf['dropout']))
        elif(iterNbr < nbrOfLayers):
            # hidden layers
            print(f'Add LSTM hidden Layer with {i} Nodes')
            model.add(CuDNNLSTM(i, return_sequences=True))
            if conf['useDropout']:
                model.add(Dropout(conf['dropout']))
        elif(iterNbr==nbrOfLayers):
            # output layer
            print(f'Add LSTM output Layer with {i} Nodes')
            model.add(CuDNNLSTM(i, return_sequences=False))
            model.add(Dense(output_dim=conf['predictionSize']))

# compile Model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[Calculator.ownMetrik, metrics.mean_squared_error, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error, metrics.cosine_proximity]) 
model.summary()  # Prints the summary of the Model

# train the model
model.fit(trainX, trainY, epochs=conf['numberOfEpochs'], shuffle=False, callbacks=[tensorboard])

# Make the prediction
pred = model.predict(testX)

#Save the model and configuration json
model.save(os.path.join(tensorboardLogDir, 'model.h5'))
df.to_hdf(os.path.join(tensorboardLogDir, 'dataframe.h5'), key='input')

conf.save(os.path.join(tensorboardLogDir, 'model_configuration.json'))

#Plot 
plot_model(model, to_file=os.path.join(tensorboardLogDir, 'model.png'), show_shapes=True, show_layer_names=True, rankdir='LR') #save model plot
plt = Logger.plotKeras(pred, testY)
plt.savefig(os.path.join(tensorboardLogDir, 'predicition_plot.png'))
Logger.checkMoneyMaker(pred, testY)
plt.show()
# ---------------------------------------------------------------------------------------------------------------