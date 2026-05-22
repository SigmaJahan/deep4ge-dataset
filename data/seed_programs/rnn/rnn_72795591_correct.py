import numpy as np
import os
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import models, layers
from CustomCallback import EnhancedLoggingCallback

def generate_dummy_data(num_samples):
    text_data = ["Sample text"] * num_samples
    labels = np.random.randint(0, 2, size=(num_samples,))
    return text_data, labels

def create_tfidf_vectors(text_data):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), use_idf=True, analyzer='word', max_features=1000)  
    vectors = vectorizer.fit_transform(text_data).toarray()
    return vectors

def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.LSTM(32, input_shape=input_shape, activation='relu'))  
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation='relu'))  
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main(model_name):
    try:
        text_data, labels = generate_dummy_data(num_samples=10000) 
        train_data = text_data[:8000]
        test_data = text_data[8000:]
        y_train = labels[:8000]
        y_test = labels[8000:]
        
        train_vector = create_tfidf_vectors(train_data)
        test_vector = create_tfidf_vectors(test_data)
        
        train_vector = train_vector[..., None]
        test_vector = test_vector[..., None]
        
        input_shape = train_vector.shape[1:]
        model = create_model(input_shape)
        
        callback_filename = model_name + ".csv"
        
        train_dataset = tf.data.Dataset.from_tensor_slices((train_vector, y_train)) \
            .shuffle(buffer_size=len(train_vector)) \
            .batch(32) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((test_vector, y_test)) \
            .batch(32) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(train_dataset, validation_data=test_dataset, epochs=50, verbose=1, callbacks=[enhancedLoggingCallback])
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        
        score = model.evaluate(test_dataset)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("72795591.h5")