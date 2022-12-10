# -*- coding: utf-8 -*-

import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
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
import math, time
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler



df=pd.read_csv('.../Independent Project/Data/Stock data/100days_20221030EB/00175_new.csv', header=0)
df.head()
#print(len(df))
#print(df.columns)

print("Discover what the data looks like in this 100 days")

plt.rcParams["figure.figsize"] = (28,5.5)
date = np.array(df["trade_date"].astype(str))
label = np.array(df.iloc[:,2].astype(str))
plt.scatter(date, label, s=100)
#plt.plot(date,df["close"])
plt.title("Last 100 days stock markets for Heng seng Index25")
plt.xticks(range(0,100,10))
plt.legend()
plt.show()

#Normalize the data
df=df.drop(labels="trade_date",axis=1)
scaler = MinMaxScaler(feature_range=(-1, 1))
for index in df.iloc[1:]:
  df[index] = scaler.fit_transform(df[index].values.reshape(-1,1))

def split_data_X(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    print(data.shape)
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1]   
    x_test = data[train_set_size:,:-1]

    return [x_train, x_test]

def split_data_Y(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    print(data.shape)
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    

    y_train = data[:train_set_size,0]
    y_test = data[train_set_size:,0]

    return [y_train, y_test]

lookback = 11 # choose sequence length
x_train, x_test = split_data_X(df, lookback)
df2 = df.drop(labels=['label'], axis=1) 
y_train, y_test = split_data_Y(df2, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)


x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

# Settings
input_dim = 2
hidden_dim = 32
num_layers = 2
output_dim = 1
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

# Training Data
hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []
#earlyStop=EarlyStopping(monitor="MSE",verbose=2,mode='min',patience=3)
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


stockname = "0175"
sns.set_style("darkgrid")    

date2 = np.array(df.iloc[0:173,0].astype(str))
fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)
plt.subplot(1, 2, 1)
ax = sns.lineplot(x = original.index.values, y = original[0], label="Data", color='royalblue')
ax = sns.lineplot(x = predict.index.values, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
ax.set_title('Stock price for '+str(stockname), size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (USD)", size = 14)
  

plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)
plt.show()


from sklearn.metrics import mean_squared_error

# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

# calculate root mean squared error
stock = "0175"
print("Mean Square Error for the stock: ",stockname )

trainScore = mean_absolute_percentage_error(y_train[:,0], y_train_pred[:,0])
print('Train Score: %.2f MAPE' % (trainScore))

testScore = mean_absolute_percentage_error(y_test[:,0], y_test_pred[:,0])
mean_absolute_percentage_error
print('Test Score: %.2f MAPE' % (testScore))


lstm.append(trainScore)
lstm.append(testScore)
lstm.append(training_time)

