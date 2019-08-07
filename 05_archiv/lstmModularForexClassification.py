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
import pickle
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten, CuDNNLSTM
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras import optimizers
from keras import metrics

from collectorForex import *
from calculator import *
from slicer import *
from logger import *
from config import *
from sendSlack import *
from confusionMatrix import *

logDirTensorboard = ""

def lstmModular(conf, workdir):
    # --- Config ----------------------------------------------------------------------------------------------------
    global logDirTensorboard
    MODEL_NAME = 'LSTM-Modular-Forex-Test'
    time_stamp = str(datetime.datetime.utcnow()).replace(":", "-")  # date-time to name folders and data
    logDirTensorboard = os.path.join(workdir,  f'{MODEL_NAME}_{time_stamp}')
    tensorboard = TensorBoard(log_dir=logDirTensorboard, histogram_freq=0, write_graph=True, write_images=True)
    # ---------------------------------------------------------------------------------------------------------------


    # --- Preperation -----------------------------------------------------------------------------------------------
    print('Collect currencies')
    col = CollectorForex(replace=conf['reloadData'], symbols=conf['selectedSymbols'], forexScale=conf['forexScale'])
    df = col.getBySymbol('EURUSD').df
    # ---------------------------------------------------------------------------------------------------------------


    # --- Calculations ----------------------------------------------------------------------------------------------
    if conf['useMovingAvarage']:
        print('Add Moving Average')
        df['Ask_MA_4h']  = df['Ask'].rolling(window=4 ).mean()
        df['Ask_MA_12h'] = df['Ask'].rolling(window=12).mean()
        df['Ask_MA_24h'] = df['Ask'].rolling(window=24).mean()
        df = df[['Ask', 'Ask_MA_4h', 'Ask_MA_12h', 'Ask_MA_24h']]
    else:
        df = df[['Ask', 'Bid']]
    # ---------------------------------------------------------------------------------------------------------------


    # --- Slice -----------------------------------------------------------------------------------------------------
    print('Slice Dataframe')
    df = Slicer.trim(df)
    df = Calculator.scaleRelative(df)
    if conf['debug']:
        df.plot()
        plt.show()

    train, test    = Slicer.split(df, trainsetSize=0.8)
    trainX, trainY = Slicer.sliceCategory(train, block=conf['blockSize'], predictionLength=conf['predictionLength'], numCategories=conf['numCategories'])
    testX, testY   = Slicer.sliceCategory(test , block=conf['blockSize'], predictionLength=conf['predictionLength'], numCategories=conf['numCategories'])

    # debug
    print(f"TrainData: \r\n{train[-2:]}")
    print(f"TestData:  \r\n{test[-2:] }")
    # ---------------------------------------------------------------------------------------------------------------


    # --- create model ----------------------------------------------------------------------------------------------
    model = Sequential() # basic model

    nbrOfLayers = len(conf['numNodes'])
    layer = 1
    for i in conf['numNodes']:
        if(layer == 1 and nbrOfLayers != 1):
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
            if nbrOfLayers == 1:
                model.add(CuDNNLSTM(i, input_shape=(conf['blockSize'], len(df.columns)), return_sequences=False))
            else:
                model.add(CuDNNLSTM(i, return_sequences=False))
            model.add(Dense(conf['numCategories'], activation='softmax'))
        layer+=1

    # compile Model
    optimizer = conf['optimizer']
    if optimizer == 'sgd': optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
        #metrics.mean_squared_error, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error, metrics.cosine_proximity
    ]) 
    model.summary()  # Prints the summary of the Model
    # ---------------------------------------------------------------------------------------------------------------


    # --- predict ---------------------------------------------------------------------------------------------------
    confusionMatrix = ConfusionMatrix() #custom callback for confusion matrix
    model.fit(trainX, trainY, epochs=conf['numberOfEpochs'], shuffle=False, callbacks=[tensorboard, confusionMatrix], validation_data=(testX, testY),) # train the model
    pred = model.predict(testX) # Make the prediction
    # ---------------------------------------------------------------------------------------------------------------


    # --- evaluate --------------------------------------------------------------------------------------------------
    # save the model and configuration file
    model.save(os.path.join(logDirTensorboard, 'model.h5'))
    df.to_hdf(os.path.join(logDirTensorboard, 'dataframe.h5'), key='input')
    conf.save(os.path.join(logDirTensorboard, 'config.json'))
    # plot 

    # plot_model(
    #     model, 
    #     to_file=os.path.join(logDirTensorboard, 'model.png'), 
    #     show_shapes=True, 
    #     show_layer_names=True, 
    #     rankdir='LR') #safe model plot

    #plt = Logger.plotKeras(pred, testY)

    #Compare Matrix
    compareMatrix = Calculator.compareMatrix(pred, testY)
    cplt = Logger.plotCompareMatrix(compareMatrix, predictionLength=conf['predictionLength'])
    cplt.savefig(os.path.join(logDirTensorboard, 'predicition_matrix.png'))

    cplt = Logger.plotKerasCategories(pred, testX, predictionLength=conf['predictionLength'])
    cplt.savefig(os.path.join(logDirTensorboard, 'predicition_plot.png'))
    cplt = Logger.plotKerasCategories(pred[:20*conf['predictionLength']], testX[:20*conf['predictionLength']], predictionLength=conf['predictionLength'])
    cplt.savefig(os.path.join(logDirTensorboard, 'predicition_plot_small.png'))
    #with open(os.path.join(logDirTensorboard, 'plot.h5'), 'wb') as file: pickle.dump(plt, file) 

    if conf['debug']:
        plt.show()
    else:
        SendSlack.sendText(conf.toString())
        SendSlack.sendFile(os.path.join(logDirTensorboard, 'predicition_matrix.png'), 'Prediction Matrix') 
        SendSlack.sendFile(os.path.join(logDirTensorboard, 'predicition_plot.png'), 'Prediction') 
        SendSlack.sendFile(os.path.join(logDirTensorboard, 'predicition_plot_small.png'), 'Prediction') 
    # ---------------------------------------------------------------------------------------------------------------