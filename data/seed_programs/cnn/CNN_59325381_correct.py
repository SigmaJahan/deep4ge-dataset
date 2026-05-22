import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from CustomCallback import EnhancedLoggingCallback

def create_dummy_data(num_samples, num_rows, num_cols, num_channels, num_classes):
    X = np.random.rand(num_samples, num_rows, num_cols, num_channels)
    y = np.random.randint(num_classes, size=num_samples)
    return X, y

def main(model_name):
    try:
        num_samples_train = 60000
        num_samples_test = 10000
        num_rows = 28
        num_cols = 28
        num_channels = 1
        num_classes = 10

        x_train, y_train = create_dummy_data(num_samples_train, num_rows, num_cols, num_channels, num_classes)
        x_test, y_test = create_dummy_data(num_samples_test, num_rows, num_cols, num_channels, num_classes)

        x_train = x_train.reshape(num_samples_train, num_rows, num_cols, num_channels)
        x_test = x_test.reshape(num_samples_test, num_rows, num_cols, num_channels)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = keras.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), (1, 1), padding="same", input_shape=(num_rows, num_cols, num_channels)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid"))
        model.add(keras.layers.Conv2D(32, (3, 3), (1, 1), padding="same"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid"))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(num_classes, activation="softmax"))

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(4).batch(16)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(x_train, y_train, epochs=50, batch_size=16, verbose=1, validation_data=(x_test, y_test), callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(x_test, y_test)

        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("59325381.h5")