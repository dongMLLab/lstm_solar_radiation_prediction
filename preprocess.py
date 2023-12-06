import numpy as np
import pandas as pd

def read_data(mode: str):
    if mode == "corr":
        data = pd.read_csv("combined.csv")
        data = data.drop("date.1", axis=1)
        data = data.drop("date", axis=1)
        datax = data[["total_cloud", "rain", "visibility", "humidity"]]
    
    if mode == "all":
        data = pd.read_csv("combined.csv")
        data = data.drop("date.1", axis=1)
        data = data.drop("date", axis=1)
        datax = data[["total_cloud", "rain", "visibility", "humidity", "wind_speed", "wind_direction", "pressure", "temperature"]]

    datay = data["solar_radiation"]

    print("Date read Finished with Mode: {}".format(mode))

    return datax, datay

def create_sequence(
        time_steps: int,
        features: int,
        mode: str,
    ):
    print("Start pre-processing. Create Sequence: {}".format(mode))
    datax, datay = read_data(mode)

    feature_array = datax.to_numpy()
    dv_array = datay.to_numpy()

    num_sequence = len(feature_array) - time_steps

    X = np.zeros((num_sequence, time_steps, features))
    y = np.zeros(num_sequence)

    for i in range(num_sequence):
        X[i] = feature_array[i: i+time_steps]
        y[i] = dv_array[i + time_steps]

    print("Created Sequence - time_step: {}, features: {}".format(time_steps, features))

    train_size = round(len(X) * 0.7)

    trainX = X[:train_size]
    trainy = y[:train_size]

    testX = X[train_size:]
    testy = y[train_size:]

    print("Split Data with train - test: data size: {}".format(train_size))

    return trainX, trainy, testX, testy