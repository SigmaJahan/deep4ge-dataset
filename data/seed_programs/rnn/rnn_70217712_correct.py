import numpy as np
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.optimizers import Adam
from CustomCallback import EnhancedLoggingCallback

data_length = 4
number_of_channels = 16
x_train = np.random.rand(17000, data_length, number_of_channels)
y_train = np.random.randint(0, 2, size=(17000, 1))
x_test = np.random.rand(4000, data_length, number_of_channels)
y_test = np.random.randint(0, 2, size=(4000, 1))

def load_model():
    model = Sequential()
    model.add(LSTM(5, recurrent_dropout=0.2, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.00001, seed=7),
                   activation="relu", input_shape=(data_length, number_of_channels), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=0.00001, clipvalue=1.5), loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
    return model

def main(model_name):
    try:
        model = load_model()
        callback_filename = model_name + ".csv"
        batch_size = 32
        epochs = 50 

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .shuffle(buffer_size=len(x_train)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        
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
    main('70217712.h5')