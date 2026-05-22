import tensorflow as tf
from tensorflow import keras
from keras import layers, metrics
import numpy as np
import os
from CustomCallback import EnhancedLoggingCallback

def create_model():
    model = keras.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(784,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10))
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def create_dummy_data(num_samples, num_features, num_classes):
    x_data = np.random.rand(num_samples, num_features)
    y_data = np.random.randint(num_classes, size=num_samples)
    y_data = keras.utils.to_categorical(y_data, num_classes)
    return x_data, y_data

def main(model_name):
    try:
        num_samples_train = 6000
        num_samples_test = 1000
        num_features = 784
        num_classes = 10

        x_train, y_train = create_dummy_data(num_samples_train, num_features, num_classes)
        x_test, y_test = create_dummy_data(num_samples_test, num_features, num_classes)

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
        model = create_model()
        model.summary()
        
        model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=test_dataset, callbacks=[enhancedLoggingCallback])
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        
        score = model.evaluate(test_dataset)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main('68692047.h5')