from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import os
from sklearn.model_selection import train_test_split
from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf

def generate_data(num_samples):
    X = np.random.rand(num_samples, 2) * 2 - 1  
    y = (X[:, 0] * X[:, 1] > 0).astype(int)  
    y = y.reshape(-1, 1) 
    return X, y

def main(model_name):
    try:
        X, y = generate_data(5000) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Sequential()
        model.add(Dense(2, input_dim=2, kernel_initializer='uniform', activation='sigmoid'))
        model.add(Dense(3, kernel_initializer='uniform', activation='sigmoid'))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10).batch(32)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        
        model.fit(train_dataset, epochs=50, verbose=1, validation_data=(X_test, y_test), callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(X_test, y_test)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("34311586.h5")