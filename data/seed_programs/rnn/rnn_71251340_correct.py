import tensorflow as tf
import os
from CustomCallback import EnhancedLoggingCallback

def generate_dummy_data():
    X_train = tf.random.normal((784, 300, 7))
    y_train = tf.random.normal((784, 300, 1))
    X_test = tf.random.normal((124, 300, 7))
    y_test = tf.random.normal((124, 300, 1)) 
    return X_train, y_train, X_test, y_test

def main(model_name):
    try:
        X_train, y_train, X_test, y_test = generate_dummy_data()
        batchsize = 32
        epochs = 50 

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True)))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
        
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
            .shuffle(buffer_size=len(X_train)) \
            .batch(batchsize) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
            .batch(batchsize) \
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

if __name__ == "__main__":
    main("71251340.h5")