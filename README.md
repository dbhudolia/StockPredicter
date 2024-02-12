# StockPrediction
## Implementation using LSTM
Long short term memory (LSTM) is a model that increases the memory of recurrent neural networks. Recurrent neural networks hold short term memory in that they allow earlier determining information to be employed in the current neural networks. For immediate tasks, the earlier data is used. We may not possess a list of all of the earlier information for the neural node. In RNNs, LSTMs are very widely used in Neural networks.

## Data Pre-processing
We must pre-process this data before applying stock price using LSTM. Transform the values in our data with help of the fit_transform function. Min-max scaler is used for scaling the data so that we can bring all the price values to a common scale.
```
    #Using MinMaxScaler for normalizing data between 0 & 1
    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

    #Defining test and train data sizes
    train_size = int(len(ds_scaled)*0.70)
    test_size = len(ds_scaled) - train_size

    #Splitting data between train and test
    ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]
```

## Implementation of our LSTM model:

In the next step, we create our LSTM model. We will use the Sequential model imported from Keras and required libraries are imported.
```
    #Creating LSTM model using keras
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1,activation='linear'))
    #model.summary()

    #Training model with adam optimizer and mean squared error loss function
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=40,batch_size=64)
```

## Conclusion

In this project, explored LSTM and stock price prediction using LSTM. Then visualized the opening and closing price value after using LSTM. After that created a Flask application for the model. Flask is a powerful and flexible micro web framework for Python, ideal for both small and large web projects. It provides a straightforward way to get a web application up and running, with all the features that you need to get started.

## Run the application 
### Install the requirement
Python requirements files are a great way to keep track of the Python modules. It is a simple text file that saves a list of the modules and packages required by your project. 
The pip utility is used to install, upgrade, and uninstall Python packages. It is also used to manage Python virtual environments and more.

To start, open up a terminal or a command prompt and navigate to the directory of this project. Once you are there, type the following command:
```
pip install -r requirements.txt
```
This will installs all of the modules listed in the Python requirements file into the project environment.

### Run Flask Application
The backend server Flask was created fully in Python. It is a framework made up of Python modules and packages. With its characteristics, it is a lightweight Flask application that speeds up the development of backend apps.

To start, open up a terminal or a command prompt and navigate to the directory of this project. Once you are there, type the following command:
```
FLASK_APP=app.py flask run
```
OR 

```
flask --app app run
```

