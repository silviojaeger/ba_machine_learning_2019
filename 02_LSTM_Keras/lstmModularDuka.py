import sys
sys.path.append('01_preperation')
sys.path.append('04_evaluation')

import pandas as pd
import tensorflow
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten, CuDNNLSTM
from keras.utils import plot_model
from keras import callbacks
from keras import optimizers
from keras import metrics
from keras import backend

from collectorDuka import *
from calculator import *
from slicer import *
from logger import *
from config import *
from sendSlack import *

logDirTensorboard = ""

def lstmModularDuka(conf, workdir):
    # --- Config ----------------------------------------------------------------------------------------------------
    global logDirTensorboard
    time_stamp = str(datetime.datetime.utcnow()).replace(":", "-")  # date-time to name folders and data
    logDirTensorboard = os.path.join(workdir,  f"{conf['name']}_{time_stamp}")
    # ---------------------------------------------------------------------------------------------------------------


    # --- Preperation -----------------------------------------------------------------------------------------------
    print('Preperation: collect dukacopy files')
    usedSymbols = [re.sub(r"_.*", "", sym) for sym in conf['selectedSymbols']]
    usedSymbols = list(dict.fromkeys(usedSymbols)) # filter duplicates with dict
    df = CollectorDuka(scale=conf['scale'], symbols=usedSymbols).df
    df = df.loc[conf['startdate']:'2019-05-01'] # trim end, change later to trimTail
    df = Slicer.trimHead(df)
    df = Calculator.scale(df)
    print(f"Data available from {df.index[0]} to {df.index[-1]}")
    # ---------------------------------------------------------------------------------------------------------------


    # --- Calculations ----------------------------------------------------------------------------------------------
    print('Calculations: parse custom functions: Moving Average')
    df = Calculator.parseFunctions(df, conf['selectedSymbols'])
    df = Slicer.trimHead(df) # the Moving Average drops some data
    df = Calculator.scaleRelative(df) # rescale relative to keep the relation between MA and Index
    if conf['debug']:
        df.plot()
        plt.show()
    # ---------------------------------------------------------------------------------------------------------------


    # --- Slice -----------------------------------------------------------------------------------------------------
    print('Slice: split dataframe and slice')
    df = df[conf['selectedSymbols']] # drop unused symbols
    train, test    = Slicer.split(df, trainsetSize=conf['trainsetSize'])
    trainX, trainY = Slicer.slice(train, block=conf['blockSize'], prediction=conf['predictionLength'])
    testX, testY   = Slicer.slice(test , block=conf['blockSize'], prediction=conf['predictionLength'])

    # debug
    if (conf['debug']):
        print(f"trainset: \r\n{train[-2:]}")
        print(f"testset:  \r\n{test[-2:] }")
    # ---------------------------------------------------------------------------------------------------------------


    # --- Neural Network --------------------------------------------------------------------------------------------
    print('Neural Network: create lstm model')
    model = Sequential() # basic model

    nbrOfLayers = len(conf['numNodes'])
    layer = 1
    for i in conf['numNodes']:
        if(layer == 1 and nbrOfLayers != 1):
            # input layer
            print(f'Add LSTM input Layer with {i} Nodes')
            if(conf['useGPU']): model.add(CuDNNLSTM(i, input_shape=(conf['blockSize'], len(df.columns)), return_sequences=True))
            else: model.add(LSTM(i, input_shape=(conf['blockSize'], len(df.columns)), return_sequences=True))
            if conf['dropout'] > 0:
                model.add(Dropout(conf['dropout']))
        elif(layer < nbrOfLayers):
            # hidden layers
            print(f'Add LSTM hidden Layer with {i} Nodes')
            if(conf['useGPU']): model.add(CuDNNLSTM(i, return_sequences=True)) 
            else: model.add(LSTM(i, return_sequences=True))
            if conf['dropout'] > 0:
                model.add(Dropout(conf['dropout']))
        elif(layer == nbrOfLayers):
            # output layer
            print(f'Add LSTM output Layer with {i} Nodes')
            if nbrOfLayers == 1:
                if(conf['useGPU']): model.add(CuDNNLSTM(i, input_shape=(conf['blockSize'], len(df.columns)), return_sequences=False))
                else: model.add(LSTM(i, input_shape=(conf['blockSize'], len(df.columns)), return_sequences=False))
            else:
                if(conf['useGPU']): model.add(CuDNNLSTM(i, return_sequences=False))
                else: model.add(LSTM(i, return_sequences=False))
            if conf['dropout'] > 0:
                model.add(Dropout(conf['dropout']))
            # add a dense at the end
            if conf['activation'] != None:
                model.add(Dense(1, activation=conf['activation']))
            else: 
                model.add(Dense(units = 1))  
        layer+=1

    # compile Model
    if conf['adamEpsilon'] == None: conf['adamEpsilon'] = backend.epsilon()
    optimizer = optimizers.Adam(lr=conf['adamLR'], beta_1=conf['adamBeta_1'], beta_2=conf['adamBeta_2'], epsilon=conf['adamEpsilon'], decay=conf['adamDecay'], amsgrad=conf['adamamSgrad'])
    model.compile(loss=Calculator.customLoss(conf), optimizer=optimizer, metrics=[
        #metrics.mean_squared_error, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error, metrics.cosine_proximity
    ]) 

    if (conf['debug']): model.summary()  # Prints the summary of the Model
    # ---------------------------------------------------------------------------------------------------------------


    # --- train -----------------------------------------------------------------------------------------------------
    print('Predict: learn prediction of testset')

    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcpSave = callbacks.ModelCheckpoint(os.path.join(logDirTensorboard, 'model.h5'), save_best_only=True, monitor='val_loss', mode='min')
    terminateNan = callbacks.TerminateOnNaN()
    tensorboard = callbacks.TensorBoard(log_dir=logDirTensorboard, histogram_freq=0, write_graph=True, write_images=True)
    class MoneyMakerCallback(callbacks.Callback):
        gradients  = []
        directions = []
        def __init__(self, epochInterval):
            self.epochInterval = epochInterval
        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.epochInterval == 0:
                testX = self.validation_data[0]
                testY = self.validation_data[1]
                # x_test, y_test = self.validation_data
                pred = self.model.predict(testX)
                gradient, direction = Logger.checkMoneyMaker(pred, testY, conf['predictionLength'])
                self.gradients  += [gradient]
                self.directions += [direction]
                print(f'Gradient  Match in [%]: {gradient:.2f}% , Direction Match in [%]: {direction:.2f}%')
    moneyMaker = MoneyMakerCallback(epochInterval=1)

    model.fit(trainX, trainY, epochs=conf['numberOfEpochs'], shuffle=False, callbacks=[tensorboard, earlyStopping, mcpSave, terminateNan, moneyMaker], validation_data=(testX, testY)) # train the model
    pred = model.predict(testX) # Make the prediction
    # ---------------------------------------------------------------------------------------------------------------


    # --- evaluate --------------------------------------------------------------------------------------------------
    print('Evaluate: evaluation of prediction')
    # save the model and configuration file
    df.to_hdf(os.path.join(logDirTensorboard, 'dataframe.h5'), key='input')
    conf.save(os.path.join(logDirTensorboard, 'config.json'))
    # plot prediction
    cplt = Logger.plotKeras(pred, testY, conf['predictionLength'])
    cplt.savefig(os.path.join(logDirTensorboard, 'predicition_plot_small.png'))
    # check money maker
    gradient, direction = Logger.checkMoneyMaker(pred, testY, conf['predictionLength'])
    print(f'Gradient  Match in [%]: {gradient:.2f}%')
    print(f'Direction Match in [%]: {direction:.2f}%')
    #with open(os.path.join(logDirTensorboard, 'plot.h5'), 'wb') as file: pickle.dump(plt, file) 

    if conf['debug']:
        plt.show()
    else:
        SendSlack.sendText(f"--- NEW TEST -----------------------------------\r\n" + \
            f"File: {logDirTensorboard}\r\n" + \
            f"{conf.toString()}\r\n" + \
            f"Gradient  Match [%]: {gradient:.2f}%\r\n" + \
            f"Direction Match [%]: {direction:.2f}%\r\n")
        #SendSlack.sendFile(os.path.join(logDirTensorboard, 'predicition_matrix.png'), 'Prediction Matrix') 
        #SendSlack.sendFile(os.path.join(logDirTensorboard, 'predicition_plot.png'), 'Prediction') 
        SendSlack.sendFile(os.path.join(logDirTensorboard, 'predicition_plot_small.png'), 'Prediction') 
    # ---------------------------------------------------------------------------------------------------------------