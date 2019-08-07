#   BA Aktienkurse vorhersagen mittels machine learning
#
#   Install graphviz on your system!
#   https://graphviz.gitlab.io/_pages/Download/Download_windows.html

import sys
sys.path.append('01_preperation')

import pandas as pd
import os
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.backend import eval, cast, sqrt, pow, floatx
from logger import *
from slicer import *
from calculator import *
from config import *

from collectorDuka import *
from slicer import *

def main():
    tensorboardLogDir = os.path.join(os.getcwd(), 'logs', 'ScaleGridSearchHoursP12_2019-07-18 11-50-27.810488\ScaleGridSearchHoursP12_2019-07-18 11-50-27.811490')
    conf = Config.load(os.path.join(tensorboardLogDir, 'config.json'))
    conf.print()

    # load df
    df = pd.read_hdf(os.path.join(tensorboardLogDir, 'dataframe.h5'), key='input')
    # slectedSymbols = conf['selectedSymbols'].copy() # wird in parseFunctions bearbeitet
    # usedSymbols = [re.sub(r"_.*", "", sym) for sym in slectedSymbols]
    # usedSymbols = [re.sub(r"-.*", "", sym) for sym in usedSymbols]
    # usedSymbols = list(dict.fromkeys(usedSymbols)) # filter duplicates with dict
    # df = CollectorDuka(scale=conf['scale'], symbols=usedSymbols).df
    # df = df.loc['2019-05-15':'2019-07-15']
    # df = Calculator.scale(df)
    # df = Calculator.parseFunctions(df, slectedSymbols)
    # df = Slicer.trimHead(df)
    # df = Slicer.trimTail(df)
    # df = Calculator.scaleRelative(df) # rescale relative to keep the relation between MA and Index
    # df = df[slectedSymbols] # drop unused symbols
    # df = df.set_index(pd.Index(range(df.shape[0])))

    # plot 
    print('sclice data')
    train, test    = Slicer.split(df, trainsetSize=0)
    #trainX, trainY = Slicer.slice(train, block=conf['blockSize'], prediction=conf['predictionLength'])
    testX, testY   = Slicer.sliceMixed(test, block=conf['blockSize'], prediction=conf['predictionLength'], specialTimeWindows=conf['specialTimeWindows'])
    testY  = [i[0] for i in testY] # remove unused data

    def plot (model, title, testX, testY):
        print(f"=== {title} ===")
        # calculate learning rate
        beta_1=0.9
        beta_2=0.999
        optimizer = model.optimizer
        lr = eval(optimizer.lr)
        print('LR (start): {:.6f}'.format(lr))
        if eval(optimizer.decay)>0:
            lr = eval(optimizer.lr) * (1. / (1. + eval(optimizer.decay) * optimizer.iterations))
        t = cast(optimizer.iterations, floatx()) + 1
        lr_t = lr * (sqrt(1. - pow(beta_2, t)) / (1. - pow(beta_1, t)))
        print('LR (now): {:.6f}'.format(eval(lr_t)))
        # prediction
        pred = model.predict(testX) # Make the prediction
        pred = [i[0] for i in pred] # remove unused data
        # matchcurve
        gradients = []
        directions = []
        blocksize = 100
        for i in range(len(pred)-21):
            predPart = pred[i:i+blocksize]
            testYPart = testY[i:i+blocksize]
            gradient, direction = Calculator.checkMoneyMakerMixed(predPart, testYPart, conf['predictionLength'])
            gradients += [gradient]
            directions += [direction]
            #print(f'Direction Match in [%]: {direction:.2f}% (Gradient  Match in [%]: {gradient:.2f}%)')
        #Logger.plotKeras(pred, testY, conf['predictionLength'])
        #plt.title(title, fontsize=18)
        # plot
        ax1 = plt.subplot(211)
        plt.plot(range(len(gradients)), gradients)
        plt.ylabel('Gradient Match [%]', fontsize=14)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(range(len(directions)), directions)
        plt.ylabel('Direction Match [%]', fontsize=14)
        plt.xlabel('Epoch', fontsize=14)
    plt.show() # plot all

    plot(load_model(os.path.join(tensorboardLogDir, 'model.h5'), custom_objects={'loss':Calculator.customLoss(conf)}), "Best Loss Model", testX, testY)
    #plot(load_model(os.path.join(tensorboardLogDir, 'directionModel.h5'), custom_objects={'loss':Calculator.customLoss(conf)}), "Best Direction Match Model", testX, testY)
    #plot(load_model(os.path.join(tensorboardLogDir, 'lastModel.h5'), custom_objects={'loss':Calculator.customLoss(conf)}), "Last Model", testX, testY)

if __name__ == '__main__':
    main()