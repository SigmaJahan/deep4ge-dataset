import numpy as np
import os
import tensorflow as tf
from CustomCallback import EnhancedLoggingCallback

def load_dataset(dataset_path):
    train_images = np.random.rand(100, 64, 64, 3)
    train_masks = np.random.randint(0, 2, size=(100, 64, 64))
    val_images = np.random.rand(20, 64, 64, 3)
    val_masks = np.random.randint(0, 2, size=(20, 64, 64))
    return (train_images, train_masks), (val_images, val_masks)

def build_unet(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(input_shape))
    model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(1, 1, activation='sigmoid'))
    return model

def tf_dataset(images, masks, batch):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def main(model_name):
    try:
        dataset_path = "building-segmentation"
        input_shape = (64, 64, 3)
        batch_size = 16
        lr = 1e-3

        (train_images, train_masks), (val_images, val_masks) = load_dataset(dataset_path)
        train_dataset = tf_dataset(train_images, train_masks, batch=batch_size)
        val_dataset = tf_dataset(val_images, val_masks, batch=batch_size)

        model = build_unet(input_shape)
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(lr),
            metrics=["accuracy"]
        )

        train_steps = (len(train_images) + batch_size - 1) // batch_size
        val_steps = (len(val_images) + batch_size - 1) // batch_size
        
        callback_filename = model_name + ".csv"
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=50,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=[enhancedLoggingCallback]
        )

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)

        score = model.evaluate(val_dataset)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("70848143.h5")