import sys
sys.path.append('01_preperation')

import pandas as pd
import os

from keras.models import load_model
from logger import *
from slicer import *
from calculator import *
from config import *

#---------------------FILE------------------
filePath = 'logs_archiv\Duka_Classification_2019-05-27 19-46-36.818072\Duka_Classification_2019-05-27 19-46-36.818072'
#-------------------------------------------

tensorboardLogDir = os.path.join(os.getcwd(), filePath)
model = load_model(os.path.join(tensorboardLogDir, 'model.h5'))
df = pd.read_hdf(os.path.join(tensorboardLogDir, 'dataframe.h5'), key='input')
conf = Config.load(os.path.join(tensorboardLogDir, 'config.json'))
conf.print()

# Plot 
print('sclice data')
train, test    = Slicer.split(df, trainsetSize=0.8)
#trainX, trainY = Slicer.slice(train, block=conf['blockSize'], prediction=conf['predictionLength'])
testX, testY   = Slicer.sliceCategory(test, block=conf['blockSize'], predictionLength=conf['predictionLength'], numCategories=conf['numCategories'])

pred = model.predict(testX) # Make the prediction

#Logger.plotKeras(pred, testY, conf['predictionLength']).show() # plot all
#Calculator.checkMoneyMakerClassification(pred, testY, conf['checkOverValue'])
Calculator.checkMoneyMakerClassification(pred, testY, 0.3)