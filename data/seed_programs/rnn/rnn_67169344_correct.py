import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from keras import layers
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        x_train = np.random.rand(1000, 100, 256)
        y_train = np.random.randint(2, size=(1000, 1))
        x_test = np.random.rand(200, 100, 256)
        y_test = np.random.randint(2, size=(200, 1))
        
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

        model = keras.Sequential()
        model.add(layers.RNN(layers.LSTMCell(256), return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1))
        model.add(layers.Activation(activation="sigmoid"))

        model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer="adam",
            metrics=["accuracy"]
        )

        model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=test_dataset, callbacks=[enhancedLoggingCallback])
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        
        score = model.evaluate(test_dataset)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("67169344.h5")