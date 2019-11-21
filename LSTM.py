import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.preprocessing import MinMaxScaler

# 读取文件
df = pd.read_csv('C:/Users/Administrator/Desktop/LSTM-test/test-1.csv')

# df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
# df.index = df['Date']

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Test_Time', 'Temperature'])
for i in range(0,len(data)):
    new_data['Test_Time'][i] = data['Test_Time'][i]
    new_data['Temperature'][i] = data['Temperature'][i]

#setting index
# new_data.index = new_data.Date
new_data.drop('Test_Time', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:8000,:]
valid = dataset[8000:,:]

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
train = new_data[:8000]
valid = new_data[8000:]
valid['Predictions'] = closing_price
plt.plot(train['Temperature'])
plt.plot(valid[['Temperature','Predictions']])
plt.show()
