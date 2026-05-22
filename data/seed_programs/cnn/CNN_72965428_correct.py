import numpy as np
from tensorflow import keras
import os
from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf

def create_dummy_data(num_samples_train, num_samples_test, num_rows, num_cols, num_channels, num_classes):
    x_train = np.random.rand(num_samples_train, num_rows, num_cols, num_channels)
    y_train = np.random.randint(num_classes, size=num_samples_train)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    x_test = np.random.rand(num_samples_test, num_rows, num_cols, num_channels)
    y_test = np.random.randint(num_classes, size=num_samples_test)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test

def main(model_name):
    try:
        num_samples_train = 60000
        num_samples_test = 10000
        num_rows = 28
        num_cols = 28
        num_channels = 3 
        num_classes = 10

        train_images, train_numbers, x_test, y_test = create_dummy_data(num_samples_train, num_samples_test, num_rows, num_cols, num_channels, num_classes)

        model = keras.Sequential()
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=train_images.shape[1:]))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(10))
        model.add(keras.layers.Softmax())

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_numbers)).shuffle(4).batch(32)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(train_dataset, epochs=50, validation_data=(x_test, y_test), verbose=1, callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)

        score = model.evaluate(x_test, y_test)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("72965428.h5")