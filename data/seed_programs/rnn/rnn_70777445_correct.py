import numpy as np
import tensorflow as tf
import os
from keras.layers import LSTM, Dense, BatchNormalization, Masking
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.optimizers import Nadam
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:   
        samples, timesteps, features = 128, 4, 99
        X_train = np.random.rand(samples, timesteps, features)
        Y_train = np.random.randint(0, 2, size=(samples))
        X_test = np.random.rand(samples, timesteps, features)
        Y_test = np.random.randint(0, 2, size=(samples))
        
        callback_filename = model_name + ".csv"
        batch_size = 32  
        epochs = 50  

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)) \
            .shuffle(buffer_size=len(X_train)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
        model.add(LSTM(100, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Nadam(learning_rate=0.0001)
        loss = BinaryCrossentropy(from_logits=False)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
        model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=test_dataset, callbacks=[enhancedLoggingCallback])
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        
        score = model.evaluate(test_dataset)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main('70777445.h5')