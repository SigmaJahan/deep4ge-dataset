import numpy as np
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers import Normalization
from CustomCallback import EnhancedLoggingCallback

def create_model(win=100, features=9):
    normalization_layer = Normalization(axis=-1)

    model = Sequential()
    model.add(normalization_layer)
    model.add(LSTM(128, activation='tanh', input_shape=(win, features), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def generate_dummy_data(sample_size=2315, time_steps=160, features=48):
    X = np.random.uniform(-1, 1, (sample_size, time_steps, features))
    Y = np.random.randint(0, 2, (sample_size, 1))
    return X, Y

def main(model_name):
    try:
        win, features = 160, 48
        x_train, y_train = generate_dummy_data()
        x_test, y_test = generate_dummy_data(sample_size=300)

        model = create_model(win, features)
        normalization_layer = model.layers[0]
        normalization_layer.adapt(x_train)
        
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .shuffle(buffer_size=len(x_train)) \
            .batch(16) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
            .batch(16) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(train_dataset, epochs=50, verbose=1, validation_data=test_dataset, callbacks=[enhancedLoggingCallback]) 
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        
        score = model.evaluate(test_dataset)
        return score
    except Exception as e:
        print(e)
        return 0
    
if __name__ == '__main__':
    main('71494820.h5')
