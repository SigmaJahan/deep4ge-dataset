import numpy as np
import os
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from keras import layers, callbacks
from CustomCallback import EnhancedLoggingCallback
from keras import metrics

def generate_dummy_dataset(num_samples=1000, num_features=10):
    dates = pd.date_range(start='1/1/2020', periods=num_samples)
    data = np.random.rand(num_samples, num_features)
    dataframe = pd.DataFrame(data, index=dates, columns=[f'feature_{i}' for i in range(num_features)])
    return dataframe

def process_pre(dataframe):
    return dataframe / dataframe.max()

def dataframe_to_xy(dataframe, window_size):
    X, Y = [], []
    for i in range(len(dataframe) - window_size - 1):
        X.append(dataframe.iloc[i:(i + window_size)].values)
        Y.append(dataframe.iloc[i + window_size].values)
    return np.array(X), np.array(Y)

def main(model_name):
    try:
        window_size = 40

        model = keras.models.Sequential()
        model.add(layers.InputLayer(input_shape=(window_size, 10)))
        model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), merge_mode='sum'))
        model.add(layers.Flatten())
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(10, activation='linear'))

        metric = metrics.MeanAbsolutePercentageError()
        model.compile(loss=keras.losses.MeanSquaredError(),
                      optimizer=keras.optimizers.Adam(),
                      metrics=[metric])
        model.summary()

        dataframe = generate_dummy_dataset()
        dataframe = process_pre(dataframe)

        epochs = 50
        batch_size = 16  
        x, y = dataframe_to_xy(dataframe, window_size)
        train_size = int(len(x) * 0.7)
        val_size = int(len(x) * 0.2)

        x_train, y_train = x[:train_size], y[:train_size]
        x_val, y_val = x[train_size:train_size + val_size], y[train_size:train_size + val_size]
        x_test, y_test = x[train_size + val_size:], y[train_size + val_size:]

        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .shuffle(buffer_size=train_size) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        score = model.evaluate(test_dataset)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main('73765263.h5')