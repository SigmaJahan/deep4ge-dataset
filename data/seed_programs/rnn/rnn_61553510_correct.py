import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from CustomCallback import EnhancedLoggingCallback


def main(model_name):
    try:
        MEAN = np.zeros(3)
        STD = np.ones(3)
        data = np.random.normal(MEAN, STD, size=(2500, 3))

        PAST_HISTORY = 5
        FUTURE_TARGET = 3
        STEP = 5

        def multivariate_data(dataset, target, start_index, end_index, history_size,
                              target_size, step, single_step=False):
            data, labels = [], []
            start_index = start_index + history_size
            if end_index is None:
                end_index = len(dataset) - target_size
            for i in range(start_index, end_index):
                indices = range(i-history_size, i, step)
                data.append(dataset[indices])
                labels.append(target[i:i+target_size] if not single_step else target[i+target_size])
            return np.array(data), np.array(labels)

        x_train, y_train = multivariate_data(dataset=data, target=data[:, -1],
                                             start_index=0, end_index=2000,
                                             history_size=PAST_HISTORY,
                                             target_size=FUTURE_TARGET, step=STEP)

        x_test, y_test = multivariate_data(dataset=data, target=data[:, -1],
                                           start_index=2000, end_index=None,
                                           history_size=PAST_HISTORY,
                                           target_size=FUTURE_TARGET, step=STEP)

        callback_filename = model_name + ".csv"
        batch_size = 32
        epochs = 50

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .shuffle(buffer_size=len(x_train)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model = tf.keras.Sequential()
        model.add(Bidirectional(LSTM(32, activation='relu', return_sequences=False), input_shape=(None, 3)))
        model.add(Dense(3))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.MeanSquaredError()

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=test_dataset, callbacks=[enhancedLoggingCallback])
        
        score = model.evaluate(test_dataset)
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main("61553510.h5")
