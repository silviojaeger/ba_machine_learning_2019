#   BA Aktienkurse vorhersagen mittels machine learning
#
#   Install graphviz on your system!
#   https://graphviz.gitlab.io/_pages/Download/Download_windows.html

import sys
sys.path.append('01_preperation')

import pandas as pd
import os

from keras.models import load_model
from logger import *
from slicer import *
from calculator import *
from config import *

tensorboardLogDir = os.path.join(os.getcwd(), 'logs', 'Duka_AbsolutePred_Develop_2019-06-19 12-16-22.214134\Duka_AbsolutePred_Develop_2019-06-19 12-16-22.214134')
df = pd.read_hdf(os.path.join(tensorboardLogDir, 'dataframe.h5'), key='input')
model = load_model(os.path.join(tensorboardLogDir, 'model.h5')), "Best Loss Model"
conf = Config.load(os.path.join(tensorboardLogDir, 'config.json'))
conf.print()

# Plot 
print('sclice data')
train, test    = Slicer.split(df, trainsetSize=0.9)
#trainX, trainY = Slicer.slice(train, block=conf['blockSize'], prediction=conf['predictionLength'])
testX, testY   = Slicer.sliceMixed(test, block=conf['blockSize'], prediction=conf['predictionLength'])

pred = model.predict(testX) # Make the prediction
gradient, direction = Logger.checkMoneyMaker(pred, testY, conf['predictionLength'])
print(f'Gradient  Match in [%]: {gradient:.2f}%')
print(f'Direction Match in [%]: {direction:.2f}%')
Logger.plotKeras(pred, testY, conf['predictionLength']).show() # plot all