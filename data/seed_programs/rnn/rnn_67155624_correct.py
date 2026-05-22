import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
import os
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:     
        trainX = np.random.rand(10000, 150)
        trainY = np.random.randint(2, size=(10000, 1))
        testX = np.random.rand(3000, 150)
        testY = np.random.randint(2, size=(3000, 1))
        EPOCHS = 20  
        BATCH_SIZE = 52  

        model = Sequential()
        model.add(LSTM(100, input_shape=(trainX.shape[1], 1), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        trainX_reshaped = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
        testX_reshaped = testX.reshape((testX.shape[0], testX.shape[1], 1))
        
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((trainX_reshaped, trainY)) \
            .shuffle(buffer_size=len(trainX_reshaped)) \
            .batch(BATCH_SIZE) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((testX_reshaped, testY)) \
            .batch(BATCH_SIZE) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        
        model.fit(train_dataset, epochs=EPOCHS, verbose=1, validation_data=test_dataset, callbacks=[enhancedLoggingCallback])
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        
        score = model.evaluate(test_dataset)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main("67155624.h5")