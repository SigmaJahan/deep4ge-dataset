import tensorflow as tf
from keras import datasets, layers, models
import os
import numpy as np
from CustomCallback import EnhancedLoggingCallback

def generate_dummy_data(num_samples=10000, image_shape=(32, 32, 3)):
    images = np.random.rand(num_samples, *image_shape).astype('float32')
    labels = np.random.randint(10, size=(num_samples, 1))  
    return images, labels

def main(model_name):
    try:
        train_images, train_labels = generate_dummy_data(10000, (32, 32, 3))
        test_images, test_labels = generate_dummy_data(2000, (32, 32, 3))
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        model = models.Sequential()
        model.add(layers.Input(shape=(32, 32, 3)))
        model.add(layers.Conv2D(25, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(50, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(100, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        BATCH_SIZE = 32
        LEARNING_RATE = 0.001

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(4).batch(BATCH_SIZE)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(train_dataset, epochs=50, validation_data=test_dataset, verbose=1, callbacks=[enhancedLoggingCallback])
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(test_dataset)
        return score

    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("64522751.h5")