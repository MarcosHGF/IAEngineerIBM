import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


# DATA : https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv

concrete_data = pd.read_csv('DL\concrete_data.csv')

print(concrete_data.head())

print(concrete_data.shape)

#print(concrete_data.describe())

#print(concrete_data.isnull().sum())

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

print(predictors.head())

print(target.head())

# Normalizing the data

predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

n_cols = predictors_norm.shape[1] # number of predictors

def regression_model():
    # create model 3 camadas, 9(input) 50 50 1(output)
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = regression_model()

model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

