import sys

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
#yahoo finance as data source
#pip install yfinance
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

app = Flask(__name__)

@app.get('/')
def getmethod():
    output = {'Working': 'Working'}
    return output


#creating dataset in time series for LSTM model 
#X[100,120,140,160,180] : Y[200]
def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)



@app.get('/wipro/')
def wiprogetmethod():
    stock_symbol = 'WIPRO.NS'

    #last 5 years data with interval of 1 day
    data = yf.download(tickers=stock_symbol,period='5y',interval='1d')
    opn = data[['Open']]
    ds = opn.values

    #Using MinMaxScaler for normalizing data between 0 & 1
    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

    #Defining test and train data sizes
    train_size = int(len(ds_scaled)*0.70)
    test_size = len(ds_scaled) - train_size

    #Splitting data between train and test
    ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

    #Taking 100 days price as one record for training
    time_stamp = 100
    X_train, y_train = create_ds(ds_train,time_stamp)
    X_test, y_test = create_ds(ds_test,time_stamp)

    #Reshaping data to fit into LSTM model
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    #Creating LSTM model using keras
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1,activation='linear'))
    #model.summary()

    #Training model with adam optimizer and mean squared error loss function
    model.compile(loss='mean_squared_error',optimizer='adam')
    #model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64)
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=40,batch_size=64)

    #Predicitng on train and test data
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    #Inverse transform to get actual value
    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)

    #Getting the last 100 days records
    x_point = len(ds_test) - 100
    fut_inp = ds_test[x_point:]
    fut_inp = fut_inp.reshape(1,-1)
    tmp_inp = list(fut_inp)
    fut_inp.shape

    #Creating list of the last 100 data
    tmp_inp = tmp_inp[0].tolist()

    #Predicting next 30 days price suing the current data
    #It will predict in sliding window manner (algorithm) with stride 1
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        
        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
        

    #print(lst_output)

    ds_new = ds_scaled.tolist()
    ds_new.extend(lst_output)
    final_graph = normalizer.inverse_transform(ds_new[1200:]).tolist()
    output = dict(enumerate(final_graph))
    return output


@app.post('/predict/')
def postmethod():

    stock_symbol = 'WIPRO.NS'

    # use parser and find the user's query
    parser = reqparse.RequestParser()
    parser.add_argument('stock', required=True)
    args = parser.parse_args()
    stock_symbol = args['stock']

    try:
        #last 5 years data with interval of 1 day
        data = yf.download(tickers=stock_symbol,period='5y',interval='1d')
        opn = data[['Open']]
        ds = opn.values
    except:
        output = {'Error': 'Invalid Symbol'}
        return output

    #Using MinMaxScaler for normalizing data between 0 & 1
    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

    #Defining test and train data sizes
    train_size = int(len(ds_scaled)*0.70)
    test_size = len(ds_scaled) - train_size

    #Splitting data between train and test
    ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

    #Taking 100 days price as one record for training
    time_stamp = 100
    X_train, y_train = create_ds(ds_train,time_stamp)
    X_test, y_test = create_ds(ds_test,time_stamp)

    #Reshaping data to fit into LSTM model
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    #Creating LSTM model using keras
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1,activation='linear'))
    #model.summary()

    #Training model with adam optimizer and mean squared error loss function
    model.compile(loss='mean_squared_error',optimizer='adam')
    #model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64)
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=40,batch_size=64)

    #Predicitng on train and test data
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    #Inverse transform to get actual value
    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)

    #Getting the last 100 days records
    x_point = len(ds_test) - 100
    fut_inp = ds_test[x_point:]
    fut_inp = fut_inp.reshape(1,-1)
    tmp_inp = list(fut_inp)
    fut_inp.shape

    #Creating list of the last 100 data
    tmp_inp = tmp_inp[0].tolist()

    #Predicting next 30 days price suing the current data
    #It will predict in sliding window manner (algorithm) with stride 1
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        
        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
        

    #print(lst_output)

    ds_new = ds_scaled.tolist()
    ds_new.extend(lst_output)
    final_graph = normalizer.inverse_transform(ds_new[1200:]).tolist()
    output = dict(enumerate(final_graph))
    return output


# Setup the Api resource routing here
# Route the URL to the resource
# api.add_resource(DepressionPredict, '/depression')

if __name__ == '__main__':
    app.run()