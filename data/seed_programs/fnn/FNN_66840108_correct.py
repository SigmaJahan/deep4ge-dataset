from CustomCallback import EnhancedLoggingCallback
import tensorflow as tf
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense

def main(model_name):
    try:
        data_str = """433,866,1299,1732
        421,842,1263,1684
        443,886,1329,1772
        142,284,426,568
        437,874,1311,1748
        455,910,1365,1820
        172,344,516,688
        219,438,657,876
        101,202,303,404
        289,578,867,1156
        110,220,330,440
        421,842,1263,1684
        472,944,1416,1888
        121,242,363,484
        215,430,645,860
        134,268,402,536
        488,976,1464,1952
        467,934,1401,1868
        418,836,1254,1672
        134,268,402,536
        241,482,723,964
        116,232,348,464
        395,790,1185,1580
        438,876,1314,1752
        396,792,1188,1584
        57,114,171,228
        218,436,654,872
        372,744,1116,1488
        305,610,915,1220
        462,924,1386,1848
        455,910,1365,1820
        42,84,126,168
        347,694,1041,1388
        394,788,1182,1576
        184,368,552,736
        302,604,906,1208
        326,652,978,1304
        333,666,999,1332
        335,670,1005,1340
        176,352,528,704
        168,336,504,672
        62,124,186,248
        26,52,78,104
        335,670,1005,1340"""

        data_lines = data_str.split("\n")
        dataset = np.array([list(map(int, line.split(','))) for line in data_lines])

        X = dataset[:,0:2]
        y = dataset[:,3]
        model = Sequential()
        model.add(Dense(196, input_dim=2, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        callback_filename = model_name + ".csv"
        train_dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(4).batch(16)
        enhancedLoggingCallback = EnhancedLoggingCallback(train_dataset, callback_filename)
        model.fit(X, y, epochs=50, batch_size=16, verbose=1, validation_data=(X,y), callbacks=[enhancedLoggingCallback])
        score = model.evaluate(X, y)
        model_location = os.path.join('trained_models', model_name)
        model.save(model_location)
        model.summary()  
        return score
    except Exception as e:
        print(e)
        return 0

if __name__ == "__main__":
    main("66840108.h5")