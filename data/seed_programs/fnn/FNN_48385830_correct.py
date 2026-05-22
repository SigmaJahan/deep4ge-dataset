import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import RandomNormal
import os
from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf
import numpy as np

def create_dummy_data(num_samples, img_rows, img_cols, num_classes):
    x_data = np.random.rand(num_samples, img_rows * img_cols)
    y_data = np.random.randint(num_classes, size=num_samples)
    y_data = keras.utils.to_categorical(y_data, num_classes)
    return x_data, y_data

def main(model_name):
    try:
        num_samples_train = 60000
        num_samples_test = 10000
        img_rows, img_cols = 28, 28
        num_classes = 10

        x_train, y_train = create_dummy_data(num_samples_train, img_rows, img_cols, num_classes)
        x_test, y_test = create_dummy_data(num_samples_test, img_rows, img_cols, num_classes)

        input_shape = (img_rows * img_cols,)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)

        model = Sequential()
        model.add(Dense(30,
                        activation='sigmoid',
                        input_shape=input_shape,
                        kernel_initializer=RandomNormal(stddev=1),
                        bias_initializer=RandomNormal(stddev=1)))
        model.add(Dense(10,
                        activation='softmax',
                        kernel_initializer=RandomNormal(stddev=1),
                        bias_initializer=RandomNormal(stddev=1)))

        model.summary()
        model.compile(optimizer=SGD(lr=0.3),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(4).batch(16)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(x_train,
                  y_train,
                  batch_size=16,
                  epochs=50,
                  verbose=1, validation_data=(x_test, y_test), callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(x_test, y_test)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("48385830.h5")