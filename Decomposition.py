import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.tsa.seasonal import seasonal_decompose
import csv

path_data='Battery-M-Segment101.csv'

df = pd.read_csv(path_data)#,encoding='utf-8', index_col='New_Test_Time'
# df2 = pd.read_csv('C:/Users/Administrator/Desktop/LSTM-test/test-3-2.csv')
time=df['Test_Time']
ts=df['Temperature']

freq=...  
decomposition = seasonal_decompose(ts.values,freq=freq, two_sided=True,extrapolate_trend=freq)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
decomposition.plot()

plt.show()
