import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, RepeatVector
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        n_sample = 10000
        n_sample_test = 2000
        num_features = 12
        num_steps_in = 11
        num_steps_out = 23
        X_train = np.random.uniform(0, 1, (n_sample, num_steps_in, num_features))
        y_train = np.random.randint(0, 2, (n_sample, num_steps_out, num_features))
        X_test = np.random.uniform(0, 1, (n_sample_test, num_steps_in, num_features))
        y_test = np.random.randint(0, 2, (n_sample_test, num_steps_out, num_features))
        
        callback_filename = model_name + ".csv"
        batch_size = 32  
        epochs = 50  

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
            .shuffle(buffer_size=len(X_train)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        
        model = Sequential()
        model.add(Bidirectional(LSTM(75), input_shape=(num_steps_in, num_features), merge_mode='concat'))
        model.add(RepeatVector(num_steps_out))
        model.add(Bidirectional(LSTM(50, return_sequences=True)))
        model.add(TimeDistributed(Dense(num_features, activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=test_dataset, callbacks=[enhancedLoggingCallback])
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        
        score = model.evaluate(test_dataset)
        model.summary()
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main("66907936.h5")