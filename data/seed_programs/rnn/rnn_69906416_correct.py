import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras import metrics
from CustomCallback import EnhancedLoggingCallback
from sklearn.metrics import r2_score

def create_dummy_data():
    np.random.seed(0)
    x = np.linspace(0, 50, 365)
    y = np.sin(x) + np.random.normal(scale=0.5, size=len(x))
    return y

def main(model_name):
    try:
        y = create_dummy_data()
        y = y.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        y = scaler.fit_transform(y)

        n_lookback = 60  
        n_forecast = 30  

        X, Y = [], []
        for i in range(n_lookback, len(y) - n_forecast + 1):
            X.append(y[i - n_lookback:i])
            Y.append(y[i:i + n_forecast])

        X = np.array(X)
        Y = np.array(Y)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(n_forecast))
        
        metric = metrics.MeanAbsolutePercentageError()
        callback_filename = model_name + ".csv"
        batch_size = 32  
        epochs = 50  

        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)) \
            .shuffle(buffer_size=len(X)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metric])
        model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=train_dataset, callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)

        X_ = y[-n_lookback:].reshape(1, n_lookback, 1)
        Y_ = model.predict(X_).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)

        dates = pd.date_range(start='today', periods=365, freq='D')
        df_past = pd.DataFrame(data={'Actual': y.flatten()}, index=dates)
        df_future = pd.DataFrame(data={'Forecast': np.nan}, index=dates)
        df_future.iloc[-n_forecast:, 0] = Y_.flatten()

        Y_pred = model.predict(X).reshape(-1, 1)
        r2 = r2_score(Y.flatten(), Y_pred.flatten())

        score = model.evaluate(train_dataset)
        return score, r2

    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main('69906416.h5')