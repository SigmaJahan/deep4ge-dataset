import tensorflow as tf
import os
from CustomCallback import EnhancedLoggingCallback

num_examples = 1000
input_shape = (32, 1000)
x_train = tf.random.normal(shape=(num_examples, *input_shape))
y_train = tf.random.uniform(shape=(num_examples,), minval=0, maxval=10, dtype=tf.int64)
num_classes = 10
batch_size = 32  

def main(model_name):
    try:
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .shuffle(buffer_size=len(x_train)) \
            .batch(batch_size) \
            .cache() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=input_shape))
        model.add(tf.keras.layers.LSTM(64, return_sequences=True))
        model.add(tf.keras.layers.LSTM(32))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        x_test = tf.random.normal(shape=(num_examples, *input_shape))
        y_test = tf.random.uniform(shape=(num_examples,), minval=0, maxval=10, dtype=tf.int64)
        
        model.fit(train_dataset, epochs=50, verbose=1, validation_data=(x_test, y_test), callbacks=[enhancedLoggingCallback])
        
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        
        score = model.evaluate(x_test, y_test)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == '__main__':
    main('67590787.h5')