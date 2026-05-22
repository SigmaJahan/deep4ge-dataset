import numpy as np
import os
from keras.layers import Conv1D, BatchNormalization, Dropout, MaxPooling1D, Flatten, Dense, Concatenate, Input
from keras.models import Sequential, Model
from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf

def main(model_name):
    try:
        num_samples = 1000
        num_test_samples = 200
        input_shape = (1000, 1)
        num_classes = 10
        train_vector1 = np.random.rand(num_samples, *input_shape)
        test_vector1 = np.random.rand(num_test_samples, *input_shape)

        train_labels = np.zeros((num_samples, num_classes))
        test_labels = np.zeros((num_test_samples, num_classes))

        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Conv1D(200, kernel_size=100, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Conv1D(200, kernel_size=100, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Conv1D(200, kernel_size=100, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(200, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((train_vector1, train_labels)).shuffle(4).batch(16)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        
        model.fit(train_vector1, train_labels, epochs=50, batch_size=16, verbose=1,
                  validation_data=(test_vector1, test_labels), callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(test_vector1, test_labels)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("73146829.h5")
