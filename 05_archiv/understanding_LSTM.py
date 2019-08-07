#
#   Please install graphviz on your system
#   https://graphviz.gitlab.io/_pages/Download/Download_windows.html

import sys
sys.path.append('01_preperation')

import pandas as pd
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import datetime
import graphviz
import pydot

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.utils import plot_model

from collector import *
from slicer import *


# --- Config ----------------------------------------------------------------------------------------------------
time_stamp = str(datetime.datetime.utcnow()).replace(":", "-")  # date-time to name folders and data
MODEL_NAME = 'LSTM-Test'
symbols = ['SCMN.SW', 'NOVN.SW', 'UBSG.SW']
tensorboardLogDir = f'./logs/{MODEL_NAME}_{time_stamp}'
tensorboard = TensorBoard(log_dir=tensorboardLogDir, histogram_freq=0, write_graph=True, write_images=False)

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# ---------------------------------------------------------------------------------------------------------------


# --- Model configuration ---------------------------------------------------------------------------------------
dimens              = 1             # dimensionality of the data. Since the data is 1-D this would be 1
predictionSize      = 1             # Number of time steps you look into the future.
blockSize           = 4             # Number of samples in a batch/sequence

inpNodes            = 7             # Number of Input Nodes
numNodes            = [14,7]        # Number of hidden nodes in each layer of the deep LSTM stack we're using
useDropout          = False         # choose dropout
dropout             = 0.2           # value of dropout
useFlatten          = True          # choose flatten (dimesninality reduction)

numberOfEpochs      = 15            # number of Epochs to train
# ---------------------------------------------------------------------------------------------------------------


# --- Help Functions --------------------------------------------------------------------------------------------
def plotKeras(prediction, truth):
    plt.figure(figsize=(18, 9))
    plt.plot(range(len(prediction)), prediction, label='prediction')
    plt.plot(range(len(truth)), truth, label='truth')
    plt.grid(True)
    plt.title(f'Plain Data: {symbols[1]}')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    plt.legend(['prediction', 'truth'])
    plt.savefig(f'{tensorboardLogDir}/predicition_plot.png')
    plt.show()
# ---------------------------------------------------------------------------------------------------------------


# --- Preperation -----------------------------------------------------------------------------------------------
col = Collector(replace=False)
print(f'Avalable Columns: {col.getBySymbol(symbols[1]).df.columns}')
sr1 = col.getBySymbol(symbols[1]).df['close']
# debug
#sr1.plot()
#plt.title(f'Plain Data: {symbols[1]}')
#plt.show()
# ---------------------------------------------------------------------------------------------------------------


# --- Slice -----------------------------------------------------------------------------------------------------
train, test    = Slicer.split(sr1, trainsetSize=0.8)
trainX, trainY = Slicer.slice([train], block=blockSize, prediction=predictionSize)[0]
testX, testY   = Slicer.slice([test], block=blockSize, prediction=predictionSize)[0]
print(testX)
print(testY)
# debug
print(f"TrainData: \r\n{train[:4]}\r\n")
print(f"TestData: \r\n{test[:4]}\r\n")
# ---------------------------------------------------------------------------------------------------------------


# --- create model ----------------------------------------------------------------------------------------------
model = Sequential()

# input layer
model.add(LSTM(inpNodes, input_shape=(blockSize, dimens), return_sequences=True))
if useDropout:
    model.add(Dropout(dropout))

# hidden layers
if len(numNodes) != 0: 
    for i in numNodes:
        print(f'Add LSTM Layer with {i} Nodes')
        model.add(LSTM(i, return_sequences=True))
        if useDropout:
            model.add(Dropout(dropout))

# output layer
if useFlatten:
    model.add(Flatten())
model.add(Dense(output_dim=predictionSize))


# compile Model
model.compile(optimizer='rmsprop', loss='mse')
model.summary()  # Prints the summary of the Model

# train the model
model.fit(trainX, trainY, epochs=numberOfEpochs,
          shuffle=False, callbacks=[tensorboard])


plot_model(model, to_file=f'{tensorboardLogDir}/model.png', show_shapes=True, show_layer_names=True, rankdir='LR') #safe model plot
# test the model with our testdata
pred = model.predict(testX)
plotKeras(pred, testY)
# ---------------------------------------------------------------------------------------------------------------