import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        num_wells = 98
        max_train_samples = 10000
        max_test_samples = 1000
        num_features = 8
        num_classes = 12

        X_train_padded = np.random.rand(max_train_samples, num_wells, num_features)
        X_test_padded = np.random.rand(max_test_samples, num_wells, num_features)
        y_train = np.random.randint(num_classes, size=(max_train_samples, num_wells))
        y_test = np.random.randint(num_classes, size=(max_test_samples, num_wells))

        y_train_padded = to_categorical(y_train, num_classes=num_classes)
        y_test_padded = to_categorical(y_test, num_classes=num_classes)
        
        callback_filename = model_name + ".csv"
        epochs = 50 
        batch_size =32 

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train_padded)) \
            .shuffle(buffer_size=len(X_train_padded)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test_padded, y_test_padded)) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
    main("68323793.h5")