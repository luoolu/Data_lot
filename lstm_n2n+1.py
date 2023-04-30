import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('historical_data.csv')
data = data.values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the dataset
X = []
y = []

for i in range(len(scaled_data) - 1):
    X.append(scaled_data[i])
    y.append(scaled_data[i + 1])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(5, 1), return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(5)
])

model.compile(loss='mean_squared_error', optimizer='adam')

# Reshape the data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

# Generate the N+1 time integers
last_data = scaled_data[-1].reshape(1, -1, 1)
predicted_data = model.predict(last_data)
predicted_data = scaler.inverse_transform(predicted_data)
predicted_data = np.round(predicted_data).astype(int)

print("Generated N+1 time integers: ", predicted_data)
