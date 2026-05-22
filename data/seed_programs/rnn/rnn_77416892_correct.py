from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import os
import tensorflow as tf
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        X_train = np.array([3.13, 4.12, 9.12, 2.56, 7.89, 1.23])
        y_train = np.array([1, 0, 0, 1, 0, 1])
        X_train = X_train.reshape(X_train.shape[0], 1, 1) 

        X_test = np.array([3.44, 2.33, 8.56, 3.76, 6.89, 0.98])
        y_test = np.array([0, 1, 0, 1, 0, 1])
        X_test = X_test.reshape(X_test.shape[0], 1, 1)

        lstm_layer_size_1 = 64
        lstm_layer_size_2 = 32
        optimizer = 'adam'
        epochs = 50
        batch_size = 32
        
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
            .shuffle(buffer_size=len(X_train)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)
        
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model = Sequential()
        model.add(LSTM(lstm_layer_size_1, activation='tanh', return_sequences=True, input_shape=(1, 1)))
        model.add(LSTM(lstm_layer_size_2, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=(X_test, y_test), callbacks=[enhancedLoggingCallback])
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        
        score = model.evaluate(X_test, y_test)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main("77416892.h5")