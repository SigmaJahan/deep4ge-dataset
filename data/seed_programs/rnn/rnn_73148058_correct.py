import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from CustomCallback import EnhancedLoggingCallback
import numpy as np

def generate_dummy_data(num_samples, sequence_length):
    x_train = np.random.rand(num_samples, sequence_length, 1)
    y_train = np.random.randint(0, 4, size=(num_samples,))  
    x_test = np.random.rand(num_samples, sequence_length, 1)
    y_test = np.random.randint(0, 4, size=(num_samples,))  

    x_train = (x_train - np.mean(x_train)) / np.std(x_train)
    x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    return x_train, y_train, x_test, y_test

def create_model():
    model = Sequential([
        LSTM(64, input_shape=(3, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(4, activation='softmax')  
    ])

    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[metric])
    return model

def main(model_name):
    try:
        num_samples = 100 
        sequence_length = 3
        batch_size = 4
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
    main('73148058.keras')