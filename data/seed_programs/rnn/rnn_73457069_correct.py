import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from CustomCallback import EnhancedLoggingCallback
from keras import metrics, optimizers

def generate_dummy_data(num_samples, sequence_length):
    x_train = np.random.rand(num_samples, sequence_length, 1)
    y_train = np.random.rand(num_samples, 1)
    x_test = np.random.rand(num_samples, sequence_length, 1)
    y_test = np.random.rand(num_samples, 1)

    x_train = (x_train - np.mean(x_train)) / np.std(x_train)
    y_train = (y_train - np.mean(y_train)) / np.std(y_train)
    x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    y_test = (y_test - np.mean(y_test)) / np.std(y_test)
    return x_train, y_train, x_test, y_test

def create_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(10000, 1)))
    model.add(Dense(1))
    metric = metrics.MeanAbsolutePercentageError()
    model.compile(loss="mean_absolute_error", optimizer=optimizers.Adam(learning_rate=0.005), metrics=[metric])
    return model

def main(model_name):
    try:
        num_samples = 57
        sequence_length = 10000
        batch_size = 16
        x_train, y_train, x_test, y_test = generate_dummy_data(num_samples, sequence_length)
        model = create_model()

        callback_filename = model_name + ".csv"
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .shuffle(buffer_size=len(x_train)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(train_dataset, epochs=50, validation_data=test_dataset, verbose=1, callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()

        score = model.evaluate(test_dataset)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("73457069.h5")