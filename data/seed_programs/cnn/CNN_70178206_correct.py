import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        num_samples = 1000
        image_shape = (64, 64, 3)
        num_classes = 10
        dummy_images = np.random.randint(0, 255, size=(num_samples,) + image_shape, dtype=np.uint8)
        dummy_labels = np.random.randint(0, num_classes, size=(num_samples,), dtype=np.int32)
        split_train = int(0.6 * num_samples)
        split_val = int(0.8 * num_samples)
        train_images, val_images, test_images = dummy_images[:split_train], dummy_images[split_train:split_val], dummy_images[split_val:]
        train_labels, val_labels, test_labels = dummy_labels[:split_train], dummy_labels[split_train:split_val], dummy_labels[split_val:]
        
        train_set = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        validation_set = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        test_set = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        
        def preprocess_image(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        train_set = train_set.map(preprocess_image).shuffle(buffer_size=1000).batch(32)
        validation_set = validation_set.map(preprocess_image).batch(32)
        test_set = test_set.map(preprocess_image).batch(32)

        model = keras.Sequential()
        model.add(keras.Input(shape=image_shape))
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=128, activation="relu"))
        model.add(layers.Dense(num_classes, activation="softmax"))

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        
        callback_filename = model_name + ".csv"
        enhancedLoggingCallback = EnhancedLoggingCallback(train_set, callback_filename)
        model.fit(train_set, epochs=50, validation_data=validation_set, verbose=1, callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        score = model.evaluate(test_set)
        model.summary()
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("70178206.h5")
