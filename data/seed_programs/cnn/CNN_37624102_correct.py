import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from keras.optimizers import SGD
import random
import os
from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf

def main(model_name):
    try:
        A = []
        B = []

        for j in range(100):
            npa = np.array([[1 for j in range(100)] for i in range(100)])
            A.append(npa.reshape(npa.shape[0], npa.shape[1], 1))  

        for j in range(100):
            npa = np.array([[0 for j in range(100)] for i in range(100)])
            B.append(npa.reshape(npa.shape[0], npa.shape[1], 1))  

        trainXA = []
        trainXB = []
        testXA = []
        testXB = []

        for j in range(len(A)):
            if ((j + 2) % 7) != 0:
                trainXA.append(A[j])
                trainXB.append(B[j])
            else:
                testXA.append(A[j])
                testXB.append(B[j])

        X_train = np.array(trainXA + trainXB)
        X_test = np.array(testXA + testXB)

        Y_train = np.array([[1, 0] for _ in range(len(X_train) // 2)] + [[0, 1] for _ in range(len(X_train) // 2)])
        Y_test = np.array([[1, 0] for _ in range(len(X_test) // 2)] + [[0, 1] for _ in range(len(X_test) // 2)])

        def jumblelists(C, D):
            outC = []
            outD = []
            for j in range(len(C)):
                newpos = int(random.random() * (len(outC) + 1))
                outC = outC[:newpos] + [C[j]] + outC[newpos:]
                outD = outD[:newpos] + [D[j]] + outD[newpos:]
            return np.array(outC), np.array(outD)

        X_train, Y_train = jumblelists(X_train, Y_train)
        
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(4).batch(32)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='valid', input_shape=(100, 100, 1)))  
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))

        model.add(Dense(2))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(X_train, Y_train, batch_size=32, epochs=50, verbose=1, validation_data=(X_test, Y_test), callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()
        score = model.evaluate(X_test, Y_test)
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("37624102.h5")