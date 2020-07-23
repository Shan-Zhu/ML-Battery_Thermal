import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.preprocessing import MinMaxScaler

# read data
df = pd.read_csv('LSTM-data-Battery-M-all.csv')
path_save0='...'
path_save1='...'
path_save2='...'
#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Cycle', 'Battery-M'])
for i in range(0,len(data)):
    new_data['Cycle'][i] = data['Cycle'][i]
    new_data['Battery-M'][i] = data['Battery-M'][i]

#setting index
new_data.drop('Cycle', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

#split data
split_data=int(len(data)*0.2)  
train = dataset[0:split_data,:]
valid = dataset[split_data:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(200,len(train)):
    x_train.append(scaled_data[i-200:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=100))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=2)

# predict
inputs = new_data[len(new_data) - len(valid) - 200:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(200,inputs.shape[0]):
    X_test.append(inputs[i-200:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms = np.sqrt(np.mean(np.power((valid-closing_price),2)))
print (rms)

#for plotting
train = new_data[:split_data]
valid = new_data[split_data:]
valid['Predictions'] = closing_price
plt.plot(train['Battery-M'],lw=3)#time,t
plt.plot(valid['Battery-M'],lw=3)
plt.plot(valid['Predictions'],lw=3)

train['Battery-M'].to_csv(path_save0,index=False,sep=';')
valid['Battery-M'].to_csv(path_save1,index=False,sep=';')
valid['Predictions'].to_csv(path_save2,index=False,sep=';')

plt.show()
