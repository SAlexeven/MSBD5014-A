# -*- coding: utf-8 -*-

import pandas as pd
import math, time
import torch
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from keras.layers import LSTM, Dense,Flatten, TimeDistributed
from keras.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from os.path import exists
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import random
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping



df=pd.read_csv('.../Independent Project/Data/Stock data/20220101-20221030_Index 25 for LSTM.csv', header=0)
df.head()
df=df.iloc[103:]
#print(len(df))
#print(df.columns)

print(len(df))
print("Discover what the data looks like for this year")


plt.rcParams["figure.figsize"] = (28,5.5)
"""
stock_code=[]
for index in df.iloc[0:1,1:26]:
  stock_code.append(index)
plt.plot(df[stock_code],label=index)
plt.title("Last 200 days stock markets for Heng seng Index25")
plt.legend()
plt.show()    
"""
date = np.array(df["trade_date"].astype(str))
for index in df.iloc[0:1,1:26]:
  plt.plot(date, df[index],label=index)
  plt.title("Last 100 days stock markets for Heng seng Index25")
  plt.xticks(range(0,100,10))
  plt.legend()
  plt.show()

#Normalize the data
from sklearn.preprocessing import MinMaxScaler

df=df.drop(labels="trade_date",axis=1)
scaler = MinMaxScaler(feature_range=(-1, 1))
for index in df.iloc[0:1]:
  df[index] = scaler.fit_transform(df[index].values.reshape(-1,1))

def split_data(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

lookback = 11 # choose sequence length
x_train, y_train, x_test, y_test = split_data(df, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)

"""#Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
for index in x_train[0:1]:
  df[index] = scaler.fit_transform(df[index].values.reshape(-1,1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
"""


x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

input_dim = 25
hidden_dim = 32
num_layers = 2
output_dim = 25
num_epochs = 100

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

import time
"""
hist = np.zeros((num_epochs,input_dim))
for index in range(input_dim):
  start_time = time.time()
  
  lstm = []

  for t in range(num_epochs):
    y_train_pred = model(x_train[:,index])
    loss = criterion(y_train_pred[:,:,index], y_train_lstm[:,:,index])
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t,index] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
  training_time = time.time()-start_time
  print("Training time for","index: {}".format(training_time))
"""
hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []
earlyStop=EarlyStopping(monitor="MSE",verbose=2,mode='min',patience=3)
for t in range(num_epochs):
    y_train_pred = model(x_train)#,callbacks=[earlyStop]
    

    loss = criterion(y_train_pred, y_train_lstm)
    print("Epoch ", t, "MSE: ", loss.item())
    
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))

predict.head()


sns.set_style("darkgrid")    

date2 = np.array(df.iloc[0:173,0].astype(str))
for i in range(0,25):
  fig = plt.figure()
  fig.subplots_adjust(hspace=0.2, wspace=0.2)
  plt.subplot(1, 2, 1)
  ax = sns.lineplot(x = original.index.values, y = original[i], label="Data", color='royalblue')
  ax = sns.lineplot(x = predict.index.values, y = predict[i], label="Training Prediction (LSTM)", color='tomato')
  stockname = df.iloc[:1,i].name
  ax.set_title('Stock price for '+str(stockname), size = 14, fontweight='bold')
  ax.set_xlabel("Days", size = 14)
  ax.set_ylabel("Cost (USD)", size = 14)
  
  #ax.set_xticks(range(0,173,20))
  #ax.set_xticklabels('', size=10)


  plt.subplot(1, 2, 2)
  ax = sns.lineplot(data=hist, color='royalblue')
  ax.set_xlabel("Epoch", size = 14)
  ax.set_ylabel("Loss", size = 14)
  ax.set_title("Training Loss", size = 14, fontweight='bold')
  fig.set_figheight(6)
  fig.set_figwidth(16)
  plt.show()

# Define the accuracy
def calculate_accuracy(real, predict):
    real = np.array(real) #+ 1
    predict = np.array(predict) #+ 1
    percentage = 1 - np.mean(np.absolute((real-predict) / real))
    return percentage * 100


# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

# calculate root mean squared error
for i in range(0,25):
  stockname = df.iloc[:1,i].name
  print("Mean Square Error for the stock: ",stockname )

  #trainScore = math.sqrt(mean_absolute_percentage_error(y_train[:,i], y_train_pred[:,i]))
  trainScore = mean_absolute_percentage_error(y_train[:,i], y_train_pred[:,i])
  print('Train Score: %.2f MAPE' % (trainScore))
  #testScore = math.sqrt(mean_absolute_percentage_error(y_test[:,i], y_test_pred[:,i]))
  testScore = mean_absolute_percentage_error(y_test[:,i], y_test_pred[:,i])
  mean_absolute_percentage_error
  print('Test Score: %.2f MAPE' % (testScore))
  #accuracies = accuracy_score(y_test[:,i],y_test_pred[:,i])
  #print('Accuracy: %.4f'%(np.mean(accuracies)))

  lstm.append(trainScore)
  lstm.append(testScore)
  lstm.append(training_time)