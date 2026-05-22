from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
import os

def create_dummy_data(num_samples, num_features, num_classes):
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(num_classes, size=num_samples)
    return X, y

def main(model_name):
    try:
        num_samples = 60000
        num_features = 28 * 28
        num_classes = 10
        X_train, y_train = create_dummy_data(num_samples, num_features, num_classes)
        X_test, y_test = create_dummy_data(10000, num_features, num_classes)

        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255
        input_shape = (28, 28, 1)
        n_classes = 10
        batch_size = 32

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(learning_rate=0.05), metrics=["accuracy"])
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(4).batch(batch_size)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        
        model.fit(X_train, y_train, batch_size=batch_size, epochs=50, verbose=1, validation_data=(X_test, y_test), callbacks=[enhancedLoggingCallback])

        score = model.evaluate(X_test, y_test)
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)

        return score

    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("52448552.h5")