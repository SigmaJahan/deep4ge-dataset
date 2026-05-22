import numpy as np
import tensorflow as tf
print(tf.__version__)
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Embedding, Dropout
from keras.regularizers import l2
from CustomCallback import EnhancedLoggingCallback


vocab_size = 5000  
maxlen = 500    
embedding_dim = 128  


np.random.seed(42)
x_train = np.random.randint(0, vocab_size, size=(2500, maxlen))
x_test = np.random.randint(0, vocab_size, size=(2500, maxlen))
y_train = np.random.randint(0, 2, size=(2500, 1))
y_test = np.random.randint(0, 2, size=(2500, 1))
x_train = np.clip(x_train, 0, vocab_size - 1)
x_test = np.clip(x_test, 0, vocab_size - 1)

def main(model_name):
    try:
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
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
        model.add(LSTM(128, return_sequences=False, dropout=0.2))
        model.add(Flatten()) 
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=test_dataset, callbacks=[enhancedLoggingCallback])
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        scores = model.evaluate(test_dataset)
        return scores

    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main('66952606.h5')
