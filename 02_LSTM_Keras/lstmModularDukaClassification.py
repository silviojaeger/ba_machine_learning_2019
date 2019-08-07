import sys
sys.path.append('01_preperation')
sys.path.append('04_evaluation')

import pandas as pd
import tensorflow
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
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
    # workdir
    if not os.path.exists(logDirTensorboard): os.makedirs(logDirTensorboard, exist_ok=True)
    # parameters
    if conf['adamEpsilon'] == None: conf['adamEpsilon'] = backend.epsilon()
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
    print('Calculations: parse custom functions: Moving Average')
    df = df.loc[conf['startdate']:]
    df = Calculator.parseFunctions(df, conf['selectedSymbols'])
    df = Slicer.trimHead(df) # the Moving Average drops some data
    df = Calculator.scaleRelative(df) # rescale relative to keep the relation between MA and Index
    df = df[conf['selectedSymbols']] # drop unused symbols
    if conf['debug']:
        df.reset_index(drop=True).plot()
        plt.show()
    # ---------------------------------------------------------------------------------------------------------------


    # --- Backup Configuration --------------------------------------------------------------------------------------
    df.to_hdf(os.path.join(logDirTensorboard, 'dataframe.h5'), key='input')
    conf.save(os.path.join(logDirTensorboard, 'config.json'))
    # ---------------------------------------------------------------------------------------------------------------


    # --- Slice -----------------------------------------------------------------------------------------------------
    print('Slice: split dataframe and slice')
    train, test    = Slicer.split(df, trainsetSize=conf['trainsetSize'])
    trainX, trainY = Slicer.sliceCategory(train, block=conf['blockSize'], predictionLength=conf['predictionLength'], numCategories=conf['numCategories'])
    testX, testY   = Slicer.sliceCategory(test , block=conf['blockSize'], predictionLength=conf['predictionLength'], numCategories=conf['numCategories'])

    # debug
    if (conf['debug']):
        print('categorical spread:')
        for i in range(conf['numCategories']):
            print(f'Spread on Cat{i}: {[list(x).index(1) for x in trainY].count(i)}')
        print()
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
            else: model.add(LSTM(i, return_sequences=True, input_shape=(conf['blockSize'], len(df.columns)), activation=conf['activation']))
            if conf['dropout'] > 0:
                model.add(Dropout(conf['dropout']))
        elif(layer < nbrOfLayers):
            # hidden layers
            print(f'Add LSTM hidden Layer with {i} Nodes')
            if(conf['useGPU']): model.add(CuDNNLSTM(i, return_sequences=True)) 
            else: model.add(LSTM(i, return_sequences=True, activation=conf['activation']))
            if conf['dropout'] > 0:
                model.add(Dropout(conf['dropout']))
        elif(layer == nbrOfLayers):
            # output layer
            print(f'Add LSTM output Layer with {i} Nodes')
            if nbrOfLayers == 1:
                if(conf['useGPU']): model.add(CuDNNLSTM(i, input_shape=(conf['blockSize'], len(df.columns)), return_sequences=False))
                else: model.add(LSTM(i, input_shape=(conf['blockSize'], len(df.columns)), return_sequences=False, activation=conf['activation']))
            else:
                if(conf['useGPU']): model.add(CuDNNLSTM(i, return_sequences=False))
                else: model.add(LSTM(i, return_sequences=False, activation=conf['activation']))
            if conf['dropout'] > 0:
                model.add(Dropout(conf['dropout']))
            model.add(Dense(conf['numCategories'], activation='softmax'))
        layer+=1

    # compile Model
    if conf['adamEpsilon'] == None: conf['adamEpsilon'] = backend.epsilon()
    optimizer = optimizers.Adam(lr=conf['adamLR'], beta_1=conf['adamBeta_1'], beta_2=conf['adamBeta_2'], epsilon=conf['adamEpsilon'], decay=conf['adamDecay'], amsgrad=conf['amsgrad'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[]) 
    #model.summary()  # Prints the summary of the Model
    # ---------------------------------------------------------------------------------------------------------------


    # --- predict ---------------------------------------------------------------------------------------------------
    print('Predict: learn prediction of testset')

    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcpSave = callbacks.ModelCheckpoint(os.path.join(logDirTensorboard, 'model.h5'), save_best_only=True, monitor='val_loss', mode='min')
    reduceLrLoss = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=conf['adamEpsilon'], mode='min')
    terminateNan = callbacks.TerminateOnNaN()
    tensorboard = callbacks.TensorBoard(log_dir=logDirTensorboard, histogram_freq=0, write_graph=True, write_images=True)
    class MoneyMakerCallback(callbacks.Callback):
        def __init__(self, epochInterval):
            self.epochInterval = epochInterval
        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.epochInterval == 0:
                testX = self.validation_data[0]
                testY = self.validation_data[1]
                pred = self.model.predict(testX)
                Calculator.checkMoneyMakerClassification(pred, testY, checkOverValue=conf['checkOverValue'])
    moneyMaker = MoneyMakerCallback(epochInterval=1)
    batch_size = conf['batchSize']
    if batch_size == -1: batch_size = trainX.shape[0]
    model.fit(trainX, trainY, batch_size=batch_size, epochs=conf['numberOfEpochs'], shuffle=conf['shuffleInput'], callbacks=[tensorboard, mcpSave, terminateNan, moneyMaker], validation_data=(testX, testY)) # train the model
    # load best model
    model = load_model(os.path.join(logDirTensorboard, 'model.h5'))
    pred = model.predict(testX) # Make the prediction
    # ---------------------------------------------------------------------------------------------------------------


    # --- evaluate --------------------------------------------------------------------------------------------------
    print('Evaluate: evaluation of prediction')
    # plot compare matrix
    compareMatrix = Calculator.compareMatrix(pred, testY)
    cplt = Logger.plotCompareMatrix(compareMatrix, predictionLength=conf['predictionLength'])
    cplt.savefig(os.path.join(logDirTensorboard, 'predicition_matrix.png'))
    # plot prediction
    Calculator.checkMoneyMakerClassification(pred, testY, checkOverValue=conf['checkOverValue'])
    #cplt = Logger.plotKerasCategories(pred, testX, predictionLength=conf['predictionLength'])
    #cplt.savefig(os.path.join(logDirTensorboard, 'predicition_plot.png'))
    cplt = Logger.plotKerasCategories(pred[:20*conf['predictionLength']], testX[:20*conf['predictionLength']], predictionLength=conf['predictionLength'])
    cplt.savefig(os.path.join(logDirTensorboard, 'predicition_plot_small.png'))
    #with open(os.path.join(logDirTensorboard, 'plot.h5'), 'wb') as file: pickle.dump(plt, file) 

    if conf['debug']:
        plt.show()
    else:
        global rigthClassPerc, rigthDirectionPerc, bestDirectionPerc, directionVerySurePerc, bestDirectionVerySurePerc
        SendSlack.sendText(
            f'--- NEW TEST -----------------------------------\r\nFile: {logDirTensorboard}\r\n{conf.toString()}' + \
            f'Right class predicted: {rigthClassPerc} %\r\n' + \
            f'Right direction predicted: {rigthDirectionPerc} %\r\n' + \
            f'Best direction prediction: {bestDirectionPerc} %\r\n' + \
            f'Right direction predicted with sureness over {conf["checkOverValue"]}: {directionVerySurePerc} %\r\n' + \
            f'Best direction predicted with sureness over {conf["checkOverValue"]}: {bestDirectionVerySurePerc} %\r\n'
        )
        SendSlack.sendFile(os.path.join(logDirTensorboard, 'predicition_matrix.png'), 'Prediction Matrix') 
        #SendSlack.sendFile(os.path.join(logDirTensorboard, 'predicition_plot.png'), 'Prediction') 
        #SendSlack.sendFile(os.path.join(logDirTensorboard, 'predicition_plot_small.png'), 'Prediction') 
    # ---------------------------------------------------------------------------------------------------------------