import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_data(mode: str):
    scaler = MinMaxScaler(feature_range=(0, 1))

    data = pd.read_csv("combined.csv")
    data = data.drop("date.1", axis=1)
    data = data.drop("date", axis=1)

    if mode == "corr":
        datax = scaler.fit_transform(data[["total_cloud", "rain", "visibility", "humidity"]])
    
    if mode == "all":
        datax = scaler.fit_transform(data[["total_cloud", "rain", "visibility", "humidity", "wind_speed", "wind_direction", "pressure", "temperature"]])

    datay = scaler.fit_transform(data["solar_radiation"].to_numpy().reshape(-1, 1))

    print("Date read Finished with Mode: {}".format(mode))

    return datax, datay

def create_sequence(
        time_steps: int,
        features: int,
        mode: str,
    ):
    print("Start pre-processing. Create Sequence: {}".format(mode))

    datax, datay = read_data(mode)

    feature_array = datax
    dv_array = datay

    num_sequence = len(feature_array) - time_steps

    X = np.zeros((num_sequence, time_steps, features))
    y = np.zeros(num_sequence)

    for i in range(num_sequence):
        X[i] = feature_array[i: i+time_steps]
        y[i] = dv_array[i + time_steps, 0]

    print("Created Sequence - time_step: {}, features: {}".format(time_steps, features))

    train_size = round(len(X) * 0.7)

    trainX = X[:train_size]
    trainy = y[:train_size]

    testX = X[train_size:]
    testy = y[train_size:]

    print("Split Data with train - test: data size: {}".format(train_size))

    return trainX, trainy, testX, testy