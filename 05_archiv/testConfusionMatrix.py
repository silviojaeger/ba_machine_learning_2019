import sys
sys.path.append('01_preperation')

import os
from calculator import *
import numpy as np
import pandas as pd
import seaborn as sn

y_true = [1.0, 0.0, 0.0, 0.0, 0.0]
y_pred = [0.1, 0.8, 0.1, 0.0, 0.0]

data = {'y_pred': y_true,
        'y_true': y_pred }

df = pd.DataFrame(data, columns=['y_pred', 'y_true'])
cm = pd.crosstab(df['y_true'], df['y_pred'], rownames=['Actual'], colnames=['Predicted'], margins = True)

[
    0  , 0  , 0  , 0  , 0
    0.8, 0  , 0  , 0  , 0
    0  , 0  , 0  , 0  , 0
    0  , 0  , 0  , 0  , 0
    0  , 0  , 0  , 0  , 0
]



Calculator.ownMetrik(y_true, y_pred)