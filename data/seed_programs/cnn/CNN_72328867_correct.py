from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf
import numpy as np
import os

def main(model_name):
    try:
        X_train = np.random.rand(100, 9000)
        Y_train = (np.sum(X_train, axis=1) > 4500).astype(int).reshape(-1, 1)
        
        X_val = np.random.rand(30, 9000)
        Y_val = (np.sum(X_val, axis=1) > 4500).astype(int).reshape(-1, 1)

        X_train_vector = X_train.reshape(-1, 9000, 1)
        X_val_vector = X_val.reshape(-1, 9000, 1)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(120, kernel_size=3, padding='valid', activation='relu', input_shape=(9000, 1)))
        model.add(tf.keras.layers.MaxPooling1D(2))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(200, activation='relu'))
        model.add(tf.keras.layers.Dense(20, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_vector, Y_train)).shuffle(4).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val_vector, Y_val)).batch(32)

        callback_filename = model_name + ".csv"
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(
            train_dataset,
            epochs=50,
            verbose=1,
            validation_data=val_dataset,
            callbacks=[enhancedLoggingCallback]
        )

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(val_dataset)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("72328867.h5")