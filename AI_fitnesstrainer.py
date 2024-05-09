# distance prediction

import numpy as np
import pandas as pd
df=pd.read_csv('C:/Users/PC/Desktop/unpaid_intern/run_internproject.csv')
x=df[['time(min)','elevation(m)','training effect',' hr max(bpm)','avg run cadence(spm)','avg vertical ratio (%)'
      ,'avg vertical oscillations (cm)','avg stride length(m)']]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)
y=df['distance(km)']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,predictions))

# prediction of heart rate using RNN

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('C:/Users/PC/Desktop/ML agorithms/BSE-BOM590111.csv')
x = df[['Open']].values

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Split the data into training and testing sets
train_samples = 1590  # Number of samples for training
x_train = x_scaled[:train_samples]
x_test = x_scaled[train_samples:]

# Prepare the training data
sequence_length = 60
x_train_processed = []
y_train = []

for i in range(sequence_length, len(x_train)):
    x_train_processed.append(x_train[i - sequence_length:i])
    y_train.append(x_train[i])

x_train_processed, y_train = np.array(x_train_processed), np.array(y_train)
print("Shape of x_train:", x_train_processed.shape)

# Prepare the test data
x_test_processed = []

for i in range(sequence_length, len(x_test)):
    x_test_processed.append(x_test[i - sequence_length:i])

x_test_processed = np.array(x_test_processed)
print("Shape of x_test:", x_test_processed.shape)

x_train = x_train_processed
x_test = x_test_processed

# Prepare the target variable
y_train = x_train_processed[:, -1]
y_test = x_test_processed[:, -1]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

rnn_model = Sequential()
rnn_model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
rnn_model.add(Dropout(0.2))
rnn_model.add(LSTM(units=50, return_sequences=True))
rnn_model.add(Dropout(0.2))
rnn_model.add(LSTM(units=50, return_sequences=True))
rnn_model.add(Dropout(0.2))
rnn_model.add(LSTM(units=50, return_sequences=False))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(x_train, y_train, epochs=50, verbose=0)

predictions = rnn_model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Compute evaluation metrics
actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(actual_values, predictions)
mae = mean_absolute_error(actual_values, predictions)
rmse=np.sqrt(mse)

print("Root mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)

plt.plot(actual_values, color='red', label='Actual')
plt.plot(predictions, color='blue', label='Predicted')
plt.title('Actual versus Predicted')
plt.legend()
plt.show()

# calories prediction

# model for calories
x=df[['time(min)', 'avg pace (min/km)', 'total ascent(m)', 'elevation(m)', 'avg heart rate(bpm)', 'stride length(m)',
       'run cadence(spm)', 'vertical ratio (%)',
       'training effect', ' hr max(bpm)', 'min elevation(m)',
       'max elevation(m)', 'best pace(min/km)', 'avg run cadence(spm)',
       'avg stride length(m)', 'avg vertical ratio (%)',
       'avg vertical oscillations (cm)', ' max run cadence spm',
       'avg ground contact time(ms)', 'Total Ascent', 'Total descent ',
       'Avg moving pace (/km)', 'moving time ','distance(km)']]
y=df[['calories']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
calerror=mean_absolute_error(y_test,predictions)
print('error for calories linear regression model-',round(calerror,2))
# model for heart rate predictions
x=df[['time(min)', 'avg pace (min/km)', 'total ascent(m)', 'elevation(m)', 'stride length(m)',
       'run cadence(spm)', 'vertical ratio (%)',
       'training effect', ' hr max(bpm)', 'min elevation(m)',
       'max elevation(m)', 'best pace(min/km)', 'avg run cadence(spm)',
       'avg stride length(m)', 'avg vertical ratio (%)',
       'avg vertical oscillations (cm)', ' max run cadence spm',
       'avg ground contact time(ms)', 'Total Ascent', 'Total descent ',
       'Avg moving pace (/km)', 'moving time ',
       'distance(km)']]
y=df[['avg heart rate(bpm)']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
heartrteerror=mean_absolute_error(y_test,predictions)
print('error for heart rate linear model is-',round(heartrteerror,2))