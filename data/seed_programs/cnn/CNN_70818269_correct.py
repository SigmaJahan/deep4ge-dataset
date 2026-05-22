import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
import os
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        num_samples = 50000
        num_classes = 10
        image_shape = (32, 32, 3)

        x_train = np.random.random((num_samples,) + image_shape).astype('float32')
        y_train = np.random.randint(num_classes, size=(num_samples, 1))
        x_test = np.random.random((10000,) + image_shape).astype('float32')
        y_test = np.random.randint(num_classes, size=(10000, 1))
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        validation_split = 0.1
        validation_samples = int(len(x_train) * validation_split)
        x_validation = x_train[:validation_samples]
        y_validation = y_train[:validation_samples]
        x_train = x_train[validation_samples:]
        y_train = y_train[validation_samples:]

        model = Sequential()
        model.add(Conv2D(input_shape=image_shape, filters=8, kernel_size=(5, 5), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(units=256, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=num_classes, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
        batch_size = 32
        callback_filename = model_name + ".csv"
        enhancedLoggingCallback = EnhancedLoggingCallback(tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(4).batch(32), callback_filename)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=50, validation_data=(x_validation, y_validation),
                  verbose=1, callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()

        score = model.evaluate(x_test, y_test)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("70818269.h5")
