import tensorflow as tf
from keras import layers
import numpy as np
import os
from CustomCallback import EnhancedLoggingCallback

def create_dummy_data(num_samples, num_features, num_classes):
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(num_classes, size=num_samples)
    return X, y

def main(model_name):
    try:
        num_samples = 7000  
        num_features = 784
        num_classes = 10
        x_train_full, y_train_full = create_dummy_data(num_samples, num_features, num_classes)
        x_test, y_test = create_dummy_data(2000, num_features, num_classes)

        x_train = x_train_full[:5000]
        y_train = y_train_full[:5000]
        x_test = x_test[:2000]
        y_test = y_test[:2000]

        nb_classes = 10
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=nb_classes)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=nb_classes)

        batch_size = 16
        alpha = 0.0001

        model = tf.keras.Sequential()
        model.add(layers.Dense(784, input_shape=(784,), activation='relu'))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(10, activation='sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=alpha),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_onehot)).shuffle(4).batch(16)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(x_train, y_train_onehot, epochs=50, batch_size=batch_size, verbose=1,
                  validation_data=(x_test, y_test_onehot), callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(x_test, y_test_onehot, verbose=0)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("55328966.h5")