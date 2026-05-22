import pandas as pd
import numpy as np
import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from CustomCallback import EnhancedLoggingCallback

def main(model_name):
    try:
        dataset_size = 200
        class_1 = np.random.uniform(low=0.2, high=0.4, size=(dataset_size,))
        class_2 = np.random.uniform(low=0.5, high=0.7, size=(dataset_size,))

        dataset = []
        for i in range(0, dataset_size, 2):
            dataset.append([class_1[i], class_1[i + 1], 1])
            dataset.append([class_2[i], class_2[i + 1], 2])

        df = pd.DataFrame(data=dataset, columns=['x', 'y', 'class'])
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        x_train = df_train.iloc[:, 0:-1].values
        y_train = df_train.iloc[:, -1]
        x_test = df_test.iloc[:, 0:-1].values
        y_test = df_test.iloc[:, -1]

        label_enc = LabelEncoder()
        y_train_encoded = label_enc.fit_transform(y_train)
        y_test_encoded = label_enc.transform(y_test)
        y_train_categorical = keras.utils.to_categorical(y_train_encoded)
        y_test_categorical = keras.utils.to_categorical(y_test_encoded)

        model = Sequential()
        model.add(Dense(units=2, activation='sigmoid', input_shape=(x_train.shape[1],)))
        model.add(Dense(units=y_train_categorical.shape[1], activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_categorical)).shuffle(len(x_train)).batch(16)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)

        model.fit(train_dataset, epochs=50, validation_data=(x_test, y_test_categorical), verbose=1, callbacks=[enhancedLoggingCallback])

        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)

        score = model.evaluate(x_test, y_test_categorical)

        sk_model = MLPClassifier(hidden_layer_sizes=(3,), activation='logistic')
        sk_model.fit(x_train, y_train)

        logreg = LogisticRegression()
        logreg.fit(x_train, y_train)
        
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("50306988.h5")