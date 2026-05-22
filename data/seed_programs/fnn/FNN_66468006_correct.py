from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf
import numpy as np
import os

def create_dummy_data(num_samples, num_rows, num_cols):
    X = np.random.rand(num_samples, num_rows, num_cols)
    y = np.random.randint(10, size=num_samples)
    return X, y

def main(model_name):
    try:
        BATCH_SIZE = 32

        num_samples_train = 60000
        num_samples_test = 10000
        num_rows = 28
        num_cols = 28

        x_train, y_train = create_dummy_data(num_samples_train, num_rows, num_cols)
        x_test, y_test = create_dummy_data(num_samples_test, num_rows, num_cols)

        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(num_rows, num_cols)))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(4).batch(BATCH_SIZE)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(x_train, y_train, epochs=50, batch_size=BATCH_SIZE, verbose=1,
                  validation_split=0.33, callbacks=[enhancedLoggingCallback])

        score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)

        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("66468006.h5")