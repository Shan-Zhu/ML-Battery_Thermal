import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Importing data
df = pd.read_csv('C:/Users/Administrator/Desktop/JPS_Response/LSTM-data.csv')
cycle=df['Cycle'][:,np.newaxis]
temperature=df['Temperature'][:,np.newaxis]

split_data=int(len(df)*0.8)  

X_train=cycle[0:split_data]
y_train=temperature[0:split_data]
X_test = cycle[split_data:]
y_test = temperature[split_data:]


from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()

from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()

from sklearn import svm
model_SVR = svm.SVR()

from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)

from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)

from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)

from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()

def try_different_method(model):
  model.fit(X_train,y_train)
  score = model.score(X_test, y_test)
  result = model.predict(X_test)
  fig=plt.figure(num=1,figsize=(20,4.5))
  plt.ylim(34, 42)
  plt.xlim(-10,1200)
  ax=plt.gca()
  bwith = 6 
  ax.spines['bottom'].set_linewidth(bwith)
  ax.spines['left'].set_linewidth(bwith)
  ax.spines['top'].set_linewidth(bwith)
  ax.spines['right'].set_linewidth(bwith)
  plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.15)
  plt.tick_params(labelsize=40,which='major',width=5,length=7) 
  plt.xlabel('Cycle',fontsize=45)   
  plt.ylabel('Temp ($^\circ$C)',fontsize=45) 
  plt.plot(X_train, y_train,lw=6, label='Train')
  plt.plot(X_test,y_test, lw=6,label='Test')
  plt.plot(X_test,result,lw=6)
  plt.show()

try_different_method(model_DecisionTreeRegressor)
try_different_method(model_LinearRegression)
try_different_method(model_SVR)
try_different_method(model_RandomForestRegressor)
try_different_method(model_AdaBoostRegressor)
try_different_method(model_GradientBoostingRegressor)
try_different_method(model_BaggingRegressor)
try_different_method(model_ExtraTreeRegressor)
