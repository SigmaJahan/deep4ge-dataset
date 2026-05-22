import numpy as np
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Dense
from CustomCallback import EnhancedLoggingCallback

seq_len = 29907
num_steps = 1100
num_classes = 5

X_train = np.random.rand(num_steps, seq_len)
X_train = np.expand_dims(X_train, -1)
y_train = np.random.randint(0, 5, size=(num_steps, 1))

X_test = np.random.rand(num_steps, seq_len)
X_test = np.expand_dims(X_test, -1)
y_test = np.random.randint(0, 5, size=(num_steps, 1))

def create_model(X_train):
    model = Sequential()
    model.add(LSTM(units=32, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main(model_name):
    try:
        callback_filename = model_name + ".csv"
        batch_size = 16 
        epochs = 50

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
            .shuffle(buffer_size=1000) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
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
    main('68109005.h5')