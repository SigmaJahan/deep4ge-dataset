import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from CustomCallback import EnhancedLoggingCallback

X_train = np.random.rand(295, 5, 18)
Y_train = np.random.rand(295, 3)
X_test = np.random.rand(100, 5, 18)
Y_test = np.random.rand(100, 3)

def create_model(X_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=X_train.shape[1:]))
    model.add(Dropout(0.1))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(50))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(model_name):
    try:
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
        
        model = create_model(X_train)
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
    main('67727890.h5')