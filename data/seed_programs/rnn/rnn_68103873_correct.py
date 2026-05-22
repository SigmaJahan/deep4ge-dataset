import tensorflow as tf
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from CustomCallback import EnhancedLoggingCallback

def generate_dummy_data(num_samples, n_sentences):
    string_data = [[f"sentence_{j}_{i}" for i in range(n_sentences)] for j in range(num_samples)]
    labels = np.random.randint(0, 2, size=(num_samples, 1))
    return string_data, labels

def preprocess_data(data, max_words=10000, max_len=10):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(np.ravel(data)) 
    sequences = tokenizer.texts_to_sequences(np.ravel(data)) 
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences.reshape(-1, data.shape[1], max_len), tokenizer

def get_lstm_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=input_shape[1]))
    model.add(tf.keras.layers.Reshape((input_shape[0], input_shape[1] * 64)))  
    model.add(tf.keras.layers.LSTM(128, return_sequences=False))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.summary()
    return model

def main(model_name):
    try:
        n_sentences = 3
        string_data, labels = generate_dummy_data(num_samples=1000, n_sentences=n_sentences)

        padded_train_data, tokenizer = preprocess_data(np.array(string_data[:800]))
        padded_test_data, _ = preprocess_data(np.array(string_data[800:]), max_words=len(tokenizer.word_index)+1)

        y_train = labels[:800]
        y_test = labels[800:]
        callback_filename = model_name + ".csv"
        
        batch_size = 32  
        epochs = 50 

        train_dataset = tf.data.Dataset.from_tensor_slices((padded_train_data, y_train)) \
            .shuffle(buffer_size=len(padded_train_data)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((padded_test_data, y_test)) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        
        model = get_lstm_model((padded_train_data.shape[1], padded_train_data.shape[2]))
        model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=test_dataset, callbacks=[enhancedLoggingCallback])
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        
        score = model.evaluate(test_dataset, verbose=0)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main('68103873.h5')