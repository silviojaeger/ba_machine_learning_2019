import sys
sys.path.append('01_preperation')
sys.path.append('04_evaluation')

import pandas as pd
import tensorflow
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

from keras.losses import mse
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten
from keras.utils import plot_model
from keras.backend import eval
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
    if conf['useGPU']: from keras.layers import CuDNNLSTM
    # tensorboard
    global logDirTensorboard
    time_stamp = str(datetime.datetime.utcnow()).replace(":", "-")  # date-time to name folders and data
    logDirTensorboard = os.path.join(workdir,  f"{conf['name']}_{time_stamp}")
    # workdir
    if not os.path.exists(logDirTensorboard): os.makedirs(logDirTensorboard, exist_ok=True)
    # parameters
    if conf['adamEpsilon'] == None: conf['adamEpsilon'] = backend.epsilon()
    conf['gradientMass'] = 1 - conf['absoluteMass'] 
    # ---------------------------------------------------------------------------------------------------------------


    # --- Preperation -----------------------------------------------------------------------------------------------
    print('Preperation: collect dukacopy files')
    slectedSymbols = conf['selectedSymbols'].copy() # wird in parseFunctions bearbeitet
    usedSymbols = [re.sub(r"_.*", "", sym) for sym in slectedSymbols]
    usedSymbols = [re.sub(r"-.*", "", sym) for sym in usedSymbols]
    usedSymbols = list(dict.fromkeys(usedSymbols)) # filter duplicates with dict
    df = CollectorDuka(scale=conf['scale'], symbols=usedSymbols).df
    # ---------------------------------------------------------------------------------------------------------------


    # --- Calculations ----------------------------------------------------------------------------------------------
    print('Calculations: parse custom functions: Moving Average, ...')
    df = df.loc[conf['startdate']:]
    df = Calculator.scale(df)
    df = Calculator.parseFunctions(df, slectedSymbols)
    df = Slicer.trimHead(df)
    df = Slicer.trimTail(df)
    df = Calculator.scaleRelative(df) # rescale relative to keep the relation between MA and Index
    df = df[slectedSymbols] # drop unused symbols

    df = df.set_index(pd.Index(range(df.shape[0])))
    if conf['debug'] and conf['showPlots']:
        df.plot()
        plt.show()
    # ---------------------------------------------------------------------------------------------------------------


    # --- Backup Configuration --------------------------------------------------------------------------------------
    df.to_hdf(os.path.join(logDirTensorboard, 'dataframe.h5'), key='input')
    conf.save(os.path.join(logDirTensorboard, 'config.json'))
    # ---------------------------------------------------------------------------------------------------------------


    # --- Slice -----------------------------------------------------------------------------------------------------
    print('Slice: split dataframe and slice')
    train, test    = Slicer.split(df, trainsetSize=conf['trainsetSize'])
    trainX, trainY = Slicer.sliceMixed(train, block=conf['blockSize'], prediction=conf['predictionLength'], specialTimeWindows=conf['specialTimeWindows'])
    testX, testY   = Slicer.sliceMixed(test , block=conf['blockSize'], prediction=conf['predictionLength'], specialTimeWindows=conf['specialTimeWindows'])

    # debug
    if conf['debug'] and conf['showPlots']:
        print(f"trainset: \r\n{train[-2:]}")
        print(f"testset:  \r\n{test[-2:] }")
        Logger.plotSlices(trainX, trainY, conf['predictionLength'])
        plt.show()
    # ---------------------------------------------------------------------------------------------------------------


    # --- Neural Network --------------------------------------------------------------------------------------------
    print('Neural Network: create lstm model')
    model = Sequential() # basic model
    nbrOfLayers = len(conf['numNodes'])
    layer = 1
    input_shape=(trainX.shape[1], trainX.shape[2])
    for i in conf['numNodes']:
        if(layer == 1 and nbrOfLayers != 1):
            # input layer
            print(f'Add LSTM input Layer with {i} Nodes')
            if(conf['useGPU']): model.add(CuDNNLSTM(i, input_shape=input_shape, return_sequences=True))
            else: model.add(LSTM(i, input_shape=input_shape, return_sequences=True, activation=conf['lstmActivationCPU']))
            if conf['dropout'] > 0:
                model.add(Dropout(conf['dropout']))
        elif(layer < nbrOfLayers):
            # hidden layers
            print(f'Add LSTM hidden Layer with {i} Nodes')
            if(conf['useGPU']): model.add(CuDNNLSTM(i, return_sequences=True)) 
            else: model.add(LSTM(i, return_sequences=True, activation=conf['lstmActivationCPU']))
            if conf['dropout'] > 0:
                model.add(Dropout(conf['dropout']))
        elif(layer == nbrOfLayers):
            # output layer
            print(f'Add LSTM output Layer with {i} Nodes')
            if nbrOfLayers == 1:
                if(conf['useGPU']): model.add(CuDNNLSTM(i, input_shape=input_shape, return_sequences=False))
                else: model.add(LSTM(i, input_shape=input_shape, return_sequences=False, activation=conf['lstmActivationCPU']))
            else:
                if(conf['useGPU']): model.add(CuDNNLSTM(i, return_sequences=False))
                else: model.add(LSTM(i, return_sequences=False, activation=conf['lstmActivationCPU']))
            if conf['dropout'] > 0:
                model.add(Dropout(conf['dropout']))
            # add a dense at the end
            # model.add(Dense(int(i/2), activation=conf['activation']))
            #if conf['dropout'] > 0:
            #    model.add(Dropout(conf['dropout']))
            model.add(Dense(trainY.shape[1], activation=conf['activation']))
        layer+=1
    
    # compile Model
    optimizer = optimizers.Adam(lr=conf['adamLR'], beta_1=conf['adamBeta_1'], beta_2=conf['adamBeta_2'], epsilon=conf['adamEpsilon'], decay=conf['adamDecay'], amsgrad=conf['adamAmsgrad'])
    model.compile(loss=Calculator.customLoss(conf), optimizer=optimizer) 
    #model.compile(loss=mse, optimizer=optimizer) 
    if (conf['debug']): model.summary()  # Prints the summary of the Model
    # ---------------------------------------------------------------------------------------------------------------


    # --- train -----------------------------------------------------------------------------------------------------
    print('Predict: learn prediction of testset')
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='min')
    mcpSave = callbacks.ModelCheckpoint(os.path.join(logDirTensorboard, 'model.h5'), save_best_only=True, monitor='val_loss', mode='min')
    reduceLrLoss = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=35, verbose=1, epsilon=conf['adamEpsilon'], mode='min')
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
                testY  = [i[0] for i in testY] 
                pred   = [i[0] for i in pred]
                gradient, direction = Calculator.checkMoneyMakerMixed(pred, testY, conf['predictionLength'])
                if len(self.directions) == 0 or direction > max(self.directions): self.model.save(os.path.join(logDirTensorboard, 'directionModel.h5'))
                if len(self.gradients)  == 0 or gradient  > max(self.gradients):  self.model.save(os.path.join(logDirTensorboard, 'gradientModel.h5'))
                self.gradients  += [gradient]
                self.directions += [direction]
                print(f'Direction Match: {direction:.2f}% (Gradients-Only Match: {gradient:.2f}%)')
    moneyMaker = MoneyMakerCallback(epochInterval=1)
    batch_size = conf['batchSize']
    if batch_size == -1:   batch_size = trainX.shape[0]
    model.fit(trainX, trainY, batch_size=batch_size, epochs=conf['numberOfEpochs'], shuffle=conf['shuffle'], callbacks=[tensorboard, mcpSave, moneyMaker], validation_data=(testX, testY)) # train the model
    model.save(os.path.join(logDirTensorboard, 'lastModel.h5'))
    # ---------------------------------------------------------------------------------------------------------------


    # --- evaluate --------------------------------------------------------------------------------------------------
    print('Evaluate: evaluation of prediction')
    # load best model
    model = load_model(os.path.join(logDirTensorboard, 'model.h5'), custom_objects={"loss" : Calculator.customLoss(conf)})
    pred = model.predict(testX) # Make the prediction
    # remove unused data -> only usable for custom loss
    trainY = [i[0] for i in trainY] 
    testY  = [i[0] for i in testY] 
    pred   = [i[0] for i in pred]
    # plot prediction
    cplt = Logger.plotKeras(pred, testY, conf['predictionLength'])
    cplt.savefig(os.path.join(logDirTensorboard, 'predicition_plot.png'))
    if conf['debug']: cplt.show()
    cplt = Logger.plotMoneyMaker(directions=moneyMaker.directions, gradients=moneyMaker.gradients)
    cplt.savefig(os.path.join(logDirTensorboard, 'predicition_plot_moneymaker.png'))
    if conf['debug']: cplt.show()
    # check money maker
    gradient, direction = Calculator.checkMoneyMakerMixed(pred, testY, conf['predictionLength'])
    print(f'Gradient  Match: {gradient:.2f}%')
    print(f'Direction Match: {direction:.2f}%')
    print(f'Best Gradient  Match: {max(moneyMaker.gradients):.2f}%')
    print(f'Best Direction Match: {max(moneyMaker.directions):.2f}%')
    print(f'Last Model Gradient  Match: {moneyMaker.gradients[-1:][0]:.2f}%')
    print(f'Last Model Direction Match: {moneyMaker.directions[-1:][0]:.2f}%')
    # log
    if not conf['debug']:
        SendSlack.sendText(f"--- NEW TEST -----------------------------------\r\n" + \
            f"File: {logDirTensorboard}\r\n" + \
            f"{conf.toString()}\r\n" + \
            f"Gradient  Match: {gradient:.2f}%\r\n" + \
            f"Direction Match: {direction:.2f}%\r\n\r\n" + \
            f"Best Gradient  Match: {max(moneyMaker.gradients):.2f}%\r\n" + \
            f"Best Direction Match: {max(moneyMaker.directions):.2f}%\r\n"+ \
            f"Last Model Gradient  Match: {moneyMaker.gradients[-1:][0]:.2f}%\r\n" + \
            f"Last Model Direction Match: {moneyMaker.directions[-1:][0]:.2f}%\r\n")
        SendSlack.sendFile(os.path.join(logDirTensorboard, 'predicition_plot.png'), 'Prediction') 
        SendSlack.sendFile(os.path.join(logDirTensorboard, 'predicition_plot_moneymaker.png'), 'Moneymaker') 
    # ---------------------------------------------------------------------------------------------------------------