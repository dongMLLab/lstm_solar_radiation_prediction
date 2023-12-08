import numpy as np
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

def read_data(mode: str, scaler):
    data = pd.read_csv("combined.csv")
    data = data.drop("date.1", axis=1)
    without_index = data.drop("date", axis=1)

    x1 = without_index[["total_cloud", "rain", "visibility", "humidity"]]
    x2 = without_index[["total_cloud", "rain", "visibility", "humidity", "wind_speed", "wind_direction", "pressure", "temperature"]]
    y = without_index["solar_radiation"]

    if mode == "corr":
        datax = scaler.fit_transform(x1)
    
    if mode == "all":
        datax = scaler.fit_transform(x2)

    datay = scaler.fit_transform(y.to_numpy().reshape(-1, 1))

    print("Date read Finished with Mode: {}".format(mode))

    return datax, datay, data

def create_sequence(
        time_steps: int,
        features: int,
        mode: str,
        scaler,
    ):
    print("Start pre-processing. Create Sequence: {}".format(mode))

    feature_array, dv_array, data = read_data(mode, scaler)

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

    date_range_values = data[train_size:len(data)-time_steps]["date"]
    testy_value_series = data[train_size: num_sequence]["solar_radiation"]
    testy_value_series.index = date_range_values

    return trainX, trainy, testX, testy, date_range_values, testy_value_series