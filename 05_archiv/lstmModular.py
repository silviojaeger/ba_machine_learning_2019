#   BA Aktienkurse vorhersagen mittels machine learning
#
#   Install graphviz on your system!
#   https://graphviz.gitlab.io/_pages/Download/Download_windows.html

import sys
sys.path.append('01_preperation')
sys.path.append('04_evaluation')

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

def lstmModular(conf, workdir):
    # --- Config ----------------------------------------------------------------------------------------------------
    MODEL_NAME = 'LSTM-Modular-Test'
    time_stamp = str(datetime.datetime.utcnow()).replace(":", "-")  # date-time to name folders and data
    logDirTensorboard = os.path.join(workdir,  f'{MODEL_NAME}_{time_stamp}')
    tensorboard = TensorBoard(log_dir=logDirTensorboard, histogram_freq=0, write_graph=True, write_images=True)
    # ---------------------------------------------------------------------------------------------------------------


    # --- Preperation -----------------------------------------------------------------------------------------------
    print('Collect Companies')
    col = Collector(replace=False)
    companies = col.companies
    currencies = col.currencies

    df = pd.concat([c.df['close'] for c in companies], axis=1, sort=False)
    df.columns = [c.symbol for c in companies]

    print('Correlate Companies')
    df = Calculator.crossCorrelatingDataframes(df, conf['selectedSymbol'])
    if conf['currencySource'] != None:
        dfCurrency = col.getBySymbol(conf['currencySource']).df[conf['currenciesToCompare']]
        df = pd.concat([df, dfCurrency], axis=1, sort=False)
    # ---------------------------------------------------------------------------------------------------------------


    # --- Slice -----------------------------------------------------------------------------------------------------
    print('Slice Dataframe')
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
    model = Sequential() # basic model

    nbrOfLayers = len(conf['numNodes'])
    layer = 1
    for i in conf['numNodes']:
        if(layer == 1):
            # input layer
            print(f'Add LSTM input Layer with {i} Nodes')
            model.add(CuDNNLSTM(i, input_shape=(conf['blockSize'], len(df.columns)), return_sequences=True))
            if conf['dropout'] > 0:
                model.add(Dropout(conf['dropout']))
        elif(layer < nbrOfLayers):
            # hidden layers
            print(f'Add LSTM hidden Layer with {i} Nodes')
            model.add(CuDNNLSTM(i, return_sequences=True))
            if conf['dropout'] > 0:
                model.add(Dropout(conf['dropout']))
        elif(layer == nbrOfLayers):
            # output layer
            print(f'Add LSTM output Layer with {i} Nodes')
            model.add(CuDNNLSTM(i, return_sequences=False))
            model.add(Dense(output_dim=conf['predictionSize']))
        layer+=1

    # compile Model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[
            Calculator.ownMetrik, 
            metrics.mean_squared_error, 
            metrics.mean_absolute_error, 
            metrics.mean_absolute_percentage_error, 
            metrics.cosine_proximity]) 
    #model.summary()  # Prints the summary of the Model
    # ---------------------------------------------------------------------------------------------------------------


    # --- predict ---------------------------------------------------------------------------------------------------
    if conf['shift']:
        Calculator.shiftY(trainX, trainY)
        Calculator.shiftY(testX, testY)
    
    model.fit(trainX, trainY, epochs=conf['numberOfEpochs'], shuffle=False, callbacks=[tensorboard]) # train the model
    pred = model.predict(testX) # Make the prediction

    if conf['shift']:
        Calculator.reshiftY(trainX, trainY)
        Calculator.reshiftY(testX, testY)
        Calculator.reshiftY(testX, pred)
    # ---------------------------------------------------------------------------------------------------------------


    # --- evaluate --------------------------------------------------------------------------------------------------
    # save the model and configuration file
    model.save(os.path.join(logDirTensorboard, 'model.h5'))
    df.to_hdf(os.path.join(logDirTensorboard, 'dataframe.h5'), key='input')
    conf.save(os.path.join(logDirTensorboard, 'config.json'))
    # plot 
    plot_model(
        model, 
        to_file=os.path.join(logDirTensorboard, 'model.png'), 
        show_shapes=True, 
        show_layer_names=True, 
        rankdir='LR') #safe model plot
    plt = Logger.plotKeras(pred, testY)
    plt.savefig(os.path.join(logDirTensorboard, 'predicition_plot.png'))

    SendSlack.sendText(conf.toString())
    SendSlack.sendFile(os.path.join(logDirTensorboard, 'predicition_plot.png'), 'Prediction')    
    # ---------------------------------------------------------------------------------------------------------------