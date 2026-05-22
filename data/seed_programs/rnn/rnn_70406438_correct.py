import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        vocab_size = 10000 
        embedding_dim = 100 
        maxlen = 300 
        num_samples = 1000 
        X_train = np.random.randint(0, vocab_size, size=(num_samples, maxlen))
        y_train = np.random.randint(0, 3, size=(num_samples,))
        X_test = np.random.randint(0, vocab_size, size=(int(num_samples * 0.2), maxlen))
        y_test = np.random.randint(0, 3, size=(int(num_samples * 0.2),))

        model = Sequential()
        model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.GlobalMaxPool1D())
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(4, activation='relu'))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
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
        
        model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=1, callbacks=[enhancedLoggingCallback])
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        
        single_sample = X_test[0].reshape(1, maxlen)
        model.predict(single_sample)
        score = model.evaluate(test_dataset)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main('70406438.h5')