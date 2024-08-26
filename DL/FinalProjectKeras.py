import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load the data
concrete_data = pd.read_csv('concrete_data.csv')

concrete_data.isnull().sum()

# Separate predictors and target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

# Normalize the data
predictors_norm = (predictors - predictors.mean()) / predictors.std()

# Number of predictors
n_cols = predictors_norm.shape[1]

# Define the function to create the model
def create_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# Number of repetitions
num_repeats = 50
mse_list = []

for _ in range(num_repeats):
    # Split the normalized data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=42)

    # Create and train the model
    model = create_model()
    model.fit(X_train, y_train, epochs=50, batch_size=200, verbose=0)  # verbose=0 to suppress training output

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error
    mean_square_error = mean_squared_error(y_test, y_pred)
    mse_list.append(mean_square_error)

print(mse_list)
# Report the mean and standard deviation of the mean squared errors
mean_mse = np.mean(mse_list)
std_mse = np.std(mse_list)

print(f"Mean of Mean Squared Errors: {mean_mse}")
print(f"Standard Deviation of Mean Squared Errors: {std_mse}")
