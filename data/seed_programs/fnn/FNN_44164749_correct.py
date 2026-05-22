import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import os
from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf

def main(model_name):
    try:
        np.random.seed(42)
        X_train = np.random.rand(100, 20)
        X_test = np.random.rand(25, 20)
        y_train = np.random.randint(2, size=(100, 11))
        y_test = np.random.randint(2, size=(25, 11))

        model = Sequential()
        model.add(Dense(5000, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dropout(0.1))
        model.add(Dense(600, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(y_train.shape[1], activation='sigmoid'))

        sgd = SGD(learning_rate=0.05, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(4).batch(32)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test), callbacks=[enhancedLoggingCallback])
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(X_test, y_test)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("44164749.h5")