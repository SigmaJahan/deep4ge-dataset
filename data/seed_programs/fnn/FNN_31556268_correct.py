import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import os
import tensorflow as tf
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        model = Sequential()
        model.add(Dense(4, input_dim=2, kernel_initializer="glorot_uniform"))
        model.add(Activation("sigmoid"))
        model.add(Dense(1, kernel_initializer="glorot_uniform"))
        model.add(Activation("sigmoid"))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        train_data = np.random.randint(0, 2, (100, 2))
        label = np.array([[x[0] ^ x[1]] for x in train_data])  

        x_test = np.random.randint(0, 2, (20, 2))
        y_test = np.array([[x[0] ^ x[1]] for x in x_test]) 

        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, label)).shuffle(4).batch(16)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        
        model.fit(train_data, label, epochs=50, batch_size=16, verbose=1, validation_data=(x_test, y_test), callbacks=[enhancedLoggingCallback])
        
        model_location = os.path.join('trained_models', model_name)
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')
        model.save(model_location)
        model.summary()
        score = model.evaluate(x_test, y_test)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    result = main("31556268.h5")