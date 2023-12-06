from keras.optimizers.legacy import Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

def initiate_model(learning_rate: float, sample_sequences: int, features: int):
    optimizer = Adam(learning_rate=learning_rate)

    model = Sequential()
    model.add(Input((sample_sequences, features)))
    model.add(LSTM(10, activation="relu"))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer=optimizer, )

    print(model.summary())

    return model