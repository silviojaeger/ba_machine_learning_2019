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
    x, y   = Slicer.sliceMixed(df , block=conf['blockSize'], prediction=conf['predictionLength'], specialTimeWindows=conf['specialTimeWindows'])
    # ---------------------------------------------------------------------------------------------------------------


    # --- Neural Network --------------------------------------------------------------------------------------------
    print('Neural Network: create lstm model')
    model = Sequential() # basic model
    nbrOfLayers = len(conf['numNodes'])
    layer = 1
    input_shape=(x.shape[1], x.shape[2])
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
            model.add(Dense(y.shape[1], activation=conf['activation']))
        layer+=1
    
    # compile Model
    optimizer = optimizers.Adam(lr=conf['adamLR'], beta_1=conf['adamBeta_1'], beta_2=conf['adamBeta_2'], epsilon=conf['adamEpsilon'], decay=conf['adamDecay'], amsgrad=conf['adamAmsgrad'])
    model.compile(loss=Calculator.customLoss(conf), optimizer=optimizer) 
    #model.fit(trainX, trainY, batch_size=1, epochs=conf['numberOfEpochs'], shuffle=False) # train the model
    #model.compile(loss=mse, optimizer=optimizer) 
    if (conf['debug']): model.summary()  # Prints the summary of the Model
    # ---------------------------------------------------------------------------------------------------------------


    # --- train -----------------------------------------------------------------------------------------------------
    print('Predict: Online Learning')
    trader = Trader(balance=conf['balance'])
    balanceHistory = []
    directions = []
    directionsMatchHistory1 = []
    directionsMatchHistory2 = []
    directionsMatchHistory3 = []
    directionsMatchHistoryA = []
    directionsMatch = 0
    directionsSize1 = 20
    directionsSize2 = 100
    directionsSize3 = 500
    plotSize = 100
    plotYSpacing = 0.01
    stackSize = conf['stackSize']
    line1 = None 
    line2 = None
    line3 = None
    for i in range(stackSize, x.shape[0]-1, 1):
        x0 = x[i-stackSize:i]
        x1 = x[i].reshape(1, x.shape[1], x.shape[2])
        y0 = y[i-stackSize:i]
        y1 = y[i]
        model.fit(x0, y0, batch_size=conf['batchSize'], epochs=1, shuffle=False) # test the model
        pred = model.predict(x1)
        diffTruth = y1[0] - y1[1]
        diffPred  = pred[0][0] - y1[1]
        directions += [(diffTruth>0 and diffPred>0) or (diffTruth<0 and diffPred<0)]
        directionsMatchHistory1 += [sum(directions[-directionsSize1:])/len(directions[-directionsSize1:])*100]
        directionsMatchHistory2 += [sum(directions[-directionsSize2:])/len(directions[-directionsSize2:])*100]
        directionsMatchHistory3 += [sum(directions[-directionsSize3:])/len(directions[-directionsSize3:])*100]
        directionsMatchHistoryA += [sum(directions)/len(directions)*100]
        print(f'[{i-stackSize}/{x.shape[0]-1}] Direction Match (last {directionsSize1}): {directionsMatchHistory1[-1]:.1f}%')
        print(f'[{i-stackSize}/{x.shape[0]-1}] Direction Match (last {directionsSize2}): {directionsMatchHistory2[-1]:.1f}%')
        print(f'[{i-stackSize}/{x.shape[0]-1}] Direction Match (last {directionsSize3}): {directionsMatchHistory3[-1]:.1f}%')
        print(f'[{i-stackSize}/{x.shape[0]-1}] Direction Match (last {directionsSize3}): {directionsMatchHistory3[-1]:.1f}%')
        if i-stackSize > x.shape[0]*conf["trainsetSize"]:
            # trading
            print(f'[{i-stackSize}/{x.shape[0]-1}] Balance: {trader.balance:.2f} , Open Trades: {len(trader.getTrades("long"))} long, {len(trader.getTrades("short"))} short')
            trader.updateTrades(stockValue=y1[1])
            balanceHistory += [trader.balance]
            if pred[0][0] >= y1[1]: # go long
                trader.goLong(stockValue=y1[1], size=conf['lever'], startTime=i)
            else: # go short 
                trader.goShort(stockValue=y1[1], size=conf['lever'], startTime=i)
        if conf['debug'] and (i >= plotSize) and (i-stackSize > x.shape[0]*conf["trainsetSize"]): # plot graph
            # chart
            l1xdata = list(range(len(x[i-plotSize:i+1])))
            l1ydata = list(x[i-plotSize:i+1, -1, 0])
            l2xdata = list(range(len(x[i-plotSize:i+1])+1+conf["predictionLength"]))
            l2ydata = [None for i in range(len(x[i-plotSize:i+1])-1+conf["predictionLength"])] + [y1[1]] + [pred[0, 0]]
            l3xdata = list(range(len(balanceHistory[-plotSize-1:])))
            l3ydata = balanceHistory[-plotSize-1:]
            if line1 == None:
                plt.ion()
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                ax2.set_xlabel("")
                ax1.set_ylabel("Stock Value")
                ax2.set_ylabel("Balance")
                line1, = ax1.plot(l1xdata, l1ydata, 'k-')
                line2, = ax1.plot(l2xdata, l2ydata, 'r-')
                line3, = ax2.plot(balanceHistory[-plotSize-1:], 'k-')
                ax1.grid()
                ax2.grid()
            else:
                line1.set_xdata(l1xdata)
                line1.set_ydata(l1ydata)
                line2.set_xdata(l2xdata)
                line2.set_ydata(l2ydata)
                line3.set_xdata(l3xdata)
                line3.set_ydata(l3ydata)
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(conf["slowDownChart"])
    model.save(os.path.join(logDirTensorboard, 'lastModel.h5'))
    # ---------------------------------------------------------------------------------------------------------------


    # --- evaluate --------------------------------------------------------------------------------------------------
    print('Evaluate: evaluation of prediction')
    print(f'Direction Match (last 100): {directionsMatch:.2f}%')
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
    ax1.plot(range(len(directionsMatchHistory1[directionsSize1:])), directionsMatchHistory1[directionsSize1:])
    ax2.plot(range(len(directionsMatchHistory2[directionsSize2:])), directionsMatchHistory2[directionsSize2:])
    ax3.plot(range(len(directionsMatchHistory3[directionsSize3:])), directionsMatchHistory3[directionsSize3:])
    ax4.plot(range(len(directionsMatchHistoryA[directionsSize3:])), directionsMatchHistoryA[directionsSize3:])
    ax5.plot(range(len(balanceHistory)), balanceHistory)
    ax1.set_title(f'Direction Match (last {directionsSize1})')
    ax2.set_title(f'Direction Match (last {directionsSize2})')
    ax3.set_title(f'Direction Match (last {directionsSize3})')
    ax4.set_title(f'Direction Match (All)')
    ax5.set_title(f'Balance')
    ax2.set_ylabel('Direction Match [%]', fontsize=14)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    ax5.grid(True)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.savefig(os.path.join(logDirTensorboard, 'directionMatch.png'))
    if conf['debug']: 
        plt.show()
    # log 
    if not conf['debug']:
        SendSlack.sendText(f"--- NEW TEST -----------------------------------\r\n" + \
            f"File: {logDirTensorboard}\r\n" + \
            f"{conf.toString()}\r\n\r\n" + \
            f"[{i-stackSize}/{x.shape[0]-1}] Direction Match (last {directionsSize1}): {directionsMatchHistory1[-1]:.1f}%\r\n" + \
            f"[{i-stackSize}/{x.shape[0]-1}] Direction Match (last {directionsSize2}): {directionsMatchHistory2[-1]:.1f}%\r\n" + \
            f"[{i-stackSize}/{x.shape[0]-1}] Direction Match (last {directionsSize3}): {directionsMatchHistory3[-1]:.1f}%\r\n" + \
            f"[{i-stackSize}/{x.shape[0]-1}] Direction Match (All): {directionsMatchHistoryA[-1]:.1f}%\r\n" + \
            f"[{i-stackSize}/{x.shape[0]-1}] Balance: {trader.balance:.2f}")
        SendSlack.sendFile(os.path.join(logDirTensorboard, "directionMatch.png"), "Direction Match") 
    # ---------------------------------------------------------------------------------------------------------------