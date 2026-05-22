import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from keras import layers
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        x_train = np.random.rand(1000, 50, 128)  
        y_train = np.random.randint(3, size=(1000, 1))  
        x_test = np.random.rand(200, 50, 128)  
        y_test = np.random.randint(3, size=(200, 1))  
        
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
        model.add(layers.SimpleRNN(128, return_sequences=False, input_shape=(50, 128)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(3))
        model.add(layers.Activation("softmax"))
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
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
    main("52782432.h5")
