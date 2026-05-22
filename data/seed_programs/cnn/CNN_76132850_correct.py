import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf

def main(model_name):
    try:
        X_train = np.random.rand(100, 3, 1)
        y_train = np.random.randint(0, 10, (100,))

        X_test = np.random.rand(50, 3, 1)
        y_test = np.random.randint(0, 10, (50,))

        model = Sequential()

        model.add(Conv1D(32, 3, activation='relu', input_shape=(3, 1)))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(4).batch(32)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test), callbacks=[enhancedLoggingCallback])
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        score = model.evaluate(X_test, y_test)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("76132850.h5")