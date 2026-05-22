import numpy as np
import tensorflow as tf
import os
from keras import models, layers, losses, metrics, optimizers
from sklearn.metrics import r2_score
from CustomCallback import EnhancedLoggingCallback

def generate_dummy_data(n_samples=5000, n_timesteps=169, n_features=30):
    X = np.random.random((n_samples, n_timesteps, n_features))
    y = np.random.random((n_samples, n_timesteps, n_features))
    return X, y

def main(model_name):
    try:  
        X_train, y_train = generate_dummy_data()
        X_test, y_test = generate_dummy_data(n_samples=300)  
        metric = metrics.MeanAbsolutePercentageError()  
        opt = optimizers.RMSprop(learning_rate=0.005)
        model = models.Sequential()
        model.add(layers.LSTM(30, return_sequences=True, activation='tanh'))
        model.add(layers.Dense(30, activation='linear'))
        model.compile(loss='mse',
                      optimizer=opt,
                      metrics=[metric])
        callback_filename = model_name + ".csv"

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
            .shuffle(buffer_size=len(X_train)) \
            .batch(32) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
            .batch(32) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(train_dataset, epochs=50, verbose=1, validation_data=test_dataset, callbacks=[enhancedLoggingCallback])  
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test.flatten(), y_pred.flatten())

        score = model.evaluate(test_dataset)
        return score, r2
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main('71351646.h5')