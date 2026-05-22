from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import to_categorical

def main(model_name):
    try:
        num_samples = 60000
        num_test_samples = 10000
        img_height, img_width = 28, 28
        num_classes = 10

        X_train = np.random.random((num_samples, img_height, img_width, 1)).astype('float32')
        X_test = np.random.random((num_test_samples, img_height, img_width, 1)).astype('float32')
        y_train = np.random.randint(num_classes, size=(num_samples, 1))
        y_test = np.random.randint(num_classes, size=(num_test_samples, 1))
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        input_shape = (28, 28, 1)
        batch_size = 16
        epochs = 50

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        train_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        validation_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)
        
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((train_generator)).shuffle(4).batch(8)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        model.fit_generator(
            train_generator,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(X_test) // batch_size, verbose=1, callbacks=[enhancedLoggingCallback]
        )

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()

        score = model.evaluate(X_test, y_test, batch_size=batch_size)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("50079585.h5")