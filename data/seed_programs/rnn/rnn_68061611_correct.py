import numpy as np
import os
from keras import utils
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from CustomCallback import EnhancedLoggingCallback

np.random.seed(0)
num_samples = 10000  
num_features = 4    
num_classes = 3     

X = np.random.rand(num_samples, num_features).astype(float)
Y = np.random.randint(0, num_classes, num_samples)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_Y = utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(X, dummy_Y, test_size=0.2)
X_train = X_train.reshape(X_train.shape[0], 2, 2) 
X_test = X_test.reshape(X_test.shape[0], 2, 2)  
y_train = y_train.reshape(y_train.shape[0], num_classes)
y_test = y_test.reshape(y_test.shape[0], num_classes)

def create_nn_model():
    model = Sequential()
    model.add(LSTM(100, dropout=0.2, input_shape=(2, 2), return_sequences=False))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main(model_name):
    try:
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
        
        model = create_nn_model()
        model.summary()
        
        model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=test_dataset, callbacks=[enhancedLoggingCallback])
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        
        score = model.evaluate(test_dataset)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main('68061611.h5')
