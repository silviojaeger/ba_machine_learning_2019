import os
import pickle

tensorboardLogDir = os.path.join(os.getcwd(), 'logs', 'Modular-LSTM-Test_2019-04-04 14-44-08.536907/LSTM-Modular-Test_2019-04-04 14-44-57.492267')
plot = pickle.load(open(os.path.join(tensorboardLogDir, 'plot.h5'), 'rb'))
plot.show()