import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        np.random.seed(42)
        dates = pd.date_range(start='2020-07-09', periods=365, freq='D')
        data = np.random.randn(365, 4)
        df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'], index=dates)
        
        df_for_training = df.astype(float)
        scaler = StandardScaler()
        df_for_training_scaled = scaler.fit_transform(df_for_training)

        trainX = []
        trainY = []

        n_future = 1
        n_past = 14

        for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
            trainX.append(df_for_training_scaled[i - n_past:i, :])
            trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

        trainX, trainY = np.array(trainX), np.array(trainY)
        trainx, testx, trainy, testy = train_test_split(trainX, trainY, test_size=0.2, random_state=42)
        
        callback_filename = model_name + ".csv"
        batch_size = 32  
        epochs = 50  

        train_dataset = tf.data.Dataset.from_tensor_slices((trainx, trainy)) \
            .shuffle(buffer_size=len(trainx)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((testx, testy)) \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(trainx.shape[1], trainx.shape[2]), return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError(), MeanAbsoluteError(), RootMeanSquaredError()])
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
    main("67649606.h5")