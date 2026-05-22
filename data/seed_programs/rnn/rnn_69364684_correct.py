import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras import regularizers
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        data = np.random.rand(1000, 10, 5)
        trainX, valX = data[:800], data[800:]

        def AE_LSTM(X):
            model = Sequential()
            model.add(LSTM(8, activation="tanh", return_sequences=True, kernel_regularizer=regularizers.l2(0.00), input_shape=(X.shape[1], X.shape[2])))
            model.add(LSTM(4, activation="tanh", return_sequences=False))
            model.add(RepeatVector(X.shape[1]))
            model.add(LSTM(4, activation="tanh", return_sequences=True))
            model.add(LSTM(8, activation="tanh", return_sequences=True))
            model.add(TimeDistributed(Dense(X.shape[2])))
            return model

        model = AE_LSTM(trainX)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss="mse", metrics=['accuracy'])

        epochs = 50 
        batch_size = 32  
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainX)) \
            .shuffle(buffer_size=len(trainX)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((valX, valX)) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=val_dataset, callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()

        hidden_state_train = model.predict(trainX)[2][0]
        hidden_state_val = model.predict(valX)[2][0]
        hidden_state_train = hidden_state_train[:, ~np.all(hidden_state_train == 0.0, axis=0)]
        hidden_state_val = hidden_state_val[:, ~np.all(hidden_state_val == 0.0, axis=0)]
        
        score = model.evaluate(val_dataset)
        return score

    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main("69364684.h5")