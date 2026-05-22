import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from keras import optimizers
import os
from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf

def main(model_name):
    try:
        np.random.seed(7)
        X = np.arange(0.0, 5.0, 0.1, dtype='float32').reshape(-1, 1)
        y = 5 * np.power(X, 2) + np.power(np.random.randn(50).reshape(-1, 1), 3)
        X_test = np.arange(0.0, 5.0, 0.2, dtype='float32').reshape(-1, 1)
        y_test = 5 * np.power(X_test, 2) + np.power(np.random.randn(25).reshape(-1, 1), 3)

        model = Sequential()
        model.add(Dense(50, activation='relu', input_dim=1))
        model.add(Dense(30, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(1, activation='linear'))

        sgd = optimizers.SGD(lr=0.001)
        model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
        
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(4).batch(16)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        
        model.fit(X, y, epochs=50, batch_size=16, validation_data=(X_test,y_test), verbose=1, callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(X_test, y_test)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    print(main("48934338.h5"))