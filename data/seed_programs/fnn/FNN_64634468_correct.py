import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Sequential
from tensorflow import keras
import os
import tensorflow as tf
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        seed = 42
        X, y = make_classification(n_samples=100000, n_features=2, n_redundant=0, 
                                   n_informative=2, random_state=seed)

        df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1))
        df.columns = [*df.columns[:-1], 'Class'] 
        df['Class'] = df['Class'].astype('int')

        X = df.drop('Class', axis=1)
        y = df['Class']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
        X_val_scaled = scaler.transform(X_val.astype(np.float64))
        model = Sequential()
        model.add(Input(shape=(X_train_scaled.shape[1],)))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        opt = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).shuffle(4).batch(32)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        model.fit(X_train_scaled, y_train, batch_size=32, epochs=50, verbose=1, validation_data=(X_val_scaled, y_val), callbacks=[enhancedLoggingCallback])
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)  
        model.summary()
        score = model.evaluate(X_val_scaled, y_val)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("64634468.h5")