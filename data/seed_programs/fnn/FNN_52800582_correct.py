import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
import os
from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf

def main(model_name):
    try:
        x = np.arange(-100, 100, 0.5)
        y = x**4

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        x = x_scaler.fit_transform(x[:, None])
        y = y_scaler.fit_transform(y[:, None])
        
        x_test = np.linspace(-100, 100, 200).reshape(-1, 1)
        y_test = x_test**4
        x_test = x_scaler.transform(x_test)
        y_test = y_scaler.transform(y_test)

        model = Sequential()
        model.add(Dense(50, input_shape=(1,)))
        model.add(Activation('relu'))
        model.add(Dense(50))
        model.add(Activation('elu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(4).batch(16)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        t1 = time.time()
        for i in range(100):
            model.fit(x, y, epochs=50, batch_size=16, verbose=1, validation_data= (x_test, y_test), callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(x_test, y_test)
        
        return score
    
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("52800582.h5")