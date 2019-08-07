import pandas as pd
import tensorflow
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard
import numpy as np
import datetime

time_stamp = str(datetime.datetime.utcnow()).replace(":", "-")  # date-time to name folders and data

# Conf
MODEL_NAME = 'LSTM-Test'

data_path = 'data_temp/wtd/history/'
symbols = ['SCMN.SW', 'NOVN.SW', 'UBSG.SW']
tensorboard_log_dir = f'./logs/{MODEL_NAME}_{time_stamp}'

# Load data
print(f"{data_path}{symbols[1]}.csv")
df = pd.read_csv(f"{data_path}{symbols[1]}.csv").set_index('Date')
df['Mid'] = (df['Low']+df['High'])/2


# visualize data
def plot(df):
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), (df['Low']+df['High'])/2)
    plt.xticks(range(0, df.shape[0], 20), df.index[::20], rotation=90)
    plt.grid(True)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.show()


def plotKeras(prediction, truth):
    plt.figure(figsize=(18, 9))
    plt.plot(range(len(prediction)), prediction, label='prediction')
    plt.plot(range(len(truth)), truth, label='truth')
    plt.grid(True)
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.legend(['prediction', 'truth'])
    plt.show()


# scale the data between 0 and 1 !!NOT PERMITTED!! We also scale the test data here!
scaler = MinMaxScaler()
values = df['Mid'].values
shaped = values.reshape(values.shape[0], 1)
scaled_data = scaler.fit_transform(shaped)
print('scaled data:')
print(scaled_data)

# split train and testdata (80 : 20)
count_row = scaled_data.shape[0]  # gives number of row count
train_rows = int(count_row*0.8)

train_data = scaled_data[:train_rows]
test_data = scaled_data[train_rows:]
print("TrainData: \r\n%s\r\n" % train_data[:4])
print("TestData: \r\n%s\r\n" % test_data[:4])


def sliceData(data, blocksize):  # slices data into every day view
    X, Y = [], []
    for i in range(len(data)-blocksize-1):
        X.append(data[i:(i+blocksize), 0])  # actual block eg last week
        Y.append(data[(i+blocksize), 0])  # the next day
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    Y = np.array(Y)
    return X, Y


train_x, train_y = sliceData(train_data, 7)
test_x, test_y = sliceData(test_data, 7)

#debug
print(train_y.shape)
train_y2 = np.empty(shape=[0, 1])
for i in train_y:
    train_y2 = np.append(train_y2, [[i]], axis=0)
print("TRAIN Y2")
print(train_y2)
train_y = train_y2


def demoOfSlicedata():
    a = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]
    mean_data = np.array(a)
    print('Sliced Data:')
    print(sliceData(mean_data, 7))


demoOfSlicedata()

# Tensorboard
tensorboard = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=0,
                          write_graph=True, write_images=False)

# Model configuration--------------------------------------------------------------------------------

dimens = 1                # dimensionality of the data. Since your data is 1-D this would be 1
num_unrollings = 1          # Number of time steps you look into the future.
batch_size = 7            # Number of samples in a batch/sequence

inp_nodes = 128           # Number of Input Nodes
num_nodes = []    # Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = len(num_nodes) # number of layers        #not sure that we need this
use_dropout = True
dropout = 0.2

number_of_epochs = 5     # number of Epochs to train

# create the model------------------------------------------------------------------------------------
model = Sequential()

# input layer
model.add(LSTM(inp_nodes, input_shape=(batch_size, dimens), return_sequences=True))
if use_dropout:
    model.add(Dropout(dropout))

# hidden layers
for i in num_nodes:
    print(f'Add LSTM Layer with {i} Nodes')
    model.add(LSTM(i, return_sequences=True))
    if use_dropout:
        model.add(Dropout(dropout))

# output layer
model.add(Flatten())
model.add(Dense(output_dim=1))
model.add(Activation('sigmoid'))


# compile Model
model.compile(optimizer='adam', loss='mse', metrics=['acc'])
model.summary()  # Prints the summary of the Model

# train the model
print("TRAIN X:")
print(train_x)

print("TRAIN Y:")
print(train_y)

model.fit(train_x, train_y, epochs=number_of_epochs,
          shuffle=False, callbacks=[tensorboard])

# test the model with our testdata
pred = model.predict(test_x)
plotKeras(pred, test_y)

# Start Tenserboard via terminal with: "tensorboard --logdir ./logs"


# Todo: Fortschritt plotten
#       Evt Testdaten mit Predicition plotten