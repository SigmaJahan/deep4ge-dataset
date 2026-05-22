import numpy as np
import os
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.models import Sequential
from sklearn import preprocessing
from CustomCallback import EnhancedLoggingCallback

def create_dummy_data():
    X_train = np.random.rand(291314, 50, 8)
    Y_train = np.random.rand(291314, 5)
    X_test = np.random.rand(72829, 50, 8)
    Y_test = np.random.rand(72829, 5)
    return X_train, Y_train, X_test, Y_test

def normalize_data(X_train, X_test):
    scaler = preprocessing.MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 50*8)).reshape(-1, 50, 8)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 50*8)).reshape(-1, 50, 8)
    return X_train_scaled, X_test_scaled

def build_model(input_shape, output_units):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=input_shape, return_sequences=False))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_units))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def main(model_name):
    try:
        NFS = 5
        X_train, Y_train, X_test, Y_test = create_dummy_data()
        X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
        
        callback_filename = model_name + ".csv"
        batch_size = 32

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, Y_train)) \
            .shuffle(buffer_size=len(X_train_scaled)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test_scaled, Y_test)) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model = build_model(X_train_scaled.shape[1:], NFS)
        model.fit(train_dataset, validation_data=test_dataset, epochs=50, callbacks=[enhancedLoggingCallback])
        
        score = model.evaluate(test_dataset)
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    result = main("51971180.h5")