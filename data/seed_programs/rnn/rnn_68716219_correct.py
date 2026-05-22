import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow import keras
from CustomCallback import EnhancedLoggingCallback

Text_X = ['A positive sentence', 'A negative sentence', 'A neutral sentence'] * 100
y = np.random.randint(0, 3, size=(300,))

MAX_NB_WORDS = 3000
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="OOV")
tokenizer.fit_on_texts(Text_X)
tokenized_X = tokenizer.texts_to_sequences(Text_X)

vocab_size = len(tokenizer.word_index) + 1
embed_dim = 300
sequence_length = 30

def embedding_matrix_filteration():
    embedding_matrix = np.random.normal(0, 1, (vocab_size, embed_dim))
    return embedding_matrix

emb = embedding_matrix_filteration()

def get_model():
    embedding_layer = Embedding(
        vocab_size,
        300,
        weights=[emb],
        trainable=False,
        input_length=sequence_length
    )

    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main(model_name):
    try:
        model = get_model()
        X_padded = keras.preprocessing.sequence.pad_sequences(tokenized_X, maxlen=sequence_length)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_index, test_index = next(sss.split(X_padded, y))
        X_train, X_test = X_padded[train_index], X_padded[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = keras.utils.to_categorical(y_train, num_classes=3)
        y_test = keras.utils.to_categorical(y_test, num_classes=3)
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

        model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=test_dataset, callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        
        score = model.evaluate(test_dataset)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main('68716219.h5')