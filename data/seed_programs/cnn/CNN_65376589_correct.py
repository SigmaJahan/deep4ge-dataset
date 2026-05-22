import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import os
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        LOSS = 'sparse_categorical_crossentropy'
        OPTIMIZER = 'adam'

        model = Sequential()
        model.add(Conv2D(32, (5, 5), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(), input_shape=(32, 32, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2()))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])
        num_samples = 50000
        num_classes = 10
        X_train = np.random.random((num_samples, 32, 32, 3)).astype('float32')
        Y_train = np.random.randint(num_classes, size=(num_samples, 1))
        X_test = np.random.random((10000, 32, 32, 3)).astype('float32')
        Y_test = np.random.randint(num_classes, size=(10000, 1))

        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(4).batch(16)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(x=X_train, y=Y_train, epochs=50, batch_size=16, verbose=1, validation_data=(X_test, Y_test), callbacks=[enhancedLoggingCallback])

        score = model.evaluate(x=X_test, y=Y_test, verbose=1)
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)  
        model.summary()
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("65376589.h5")