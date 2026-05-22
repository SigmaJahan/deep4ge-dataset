from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def main(model_name):
    try:
        num_samples = 60000
        num_test_samples = 10000
        num_features = 784 
        num_classes = 2

        X_train = np.random.random((num_samples, num_features)).astype('float32')
        X_test = np.random.random((num_test_samples, num_features)).astype('float32')
        y_train = np.random.randint(num_classes, size=(num_samples, 1))
        y_test = np.random.randint(num_classes, size=(num_test_samples, 1))
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(Dense(392, kernel_initializer='normal', input_dim=num_features, activation='relu'))
        model.add(Dense(196, kernel_initializer='normal', activation='relu'))
        model.add(Dense(98, kernel_initializer='normal', activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(4).batch(32)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1, callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        score = model.evaluate(X_test, y_test)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("50201540.h5")