from lstm import initiate_model
from preprocess import create_sequence
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def train_model(model, trainX, trainy, validation_split: int, batch_size: int, epoch: int):
    early_stopping = EarlyStopping(patience=10)
    trained = model.fit(trainX, trainy, validation_split=validation_split, batch_size=batch_size, epochs=epoch, callbacks = [early_stopping])

    return trained

def visualize(predicted_values, test_value, train_history, mode: str, epoch: str, batch_size: str):
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(train_history.history['loss'], 'y', label = 'train loss')
    loss_ax.plot(train_history.history['val_loss'], 'r', label = 'val loss')
    acc_ax.plot(train_history.history['accuracy'], 'b', label = 'train accuracy')
    acc_ax.plot(train_history.history['val_accuracy'], 'g', label = 'val accuracy')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_xlabel('accuracy')

    loss_ax.legend(loc = 'upper left')
    acc_ax.legend(loc = 'lower left')

    plt.savefig("visualize/history/train_history_{}_e{}_b{}".format(mode, epoch, batch_size))
    plt.close()

    if mode == "corr":
        plt.title("Predicted with Correlated Data batch: {} (sequences)".format(batch_size))
        plt.plot(predicted_values, c="b", label="predicted")
        plt.plot(test_value, c="g", alpha=.5, label="raw")
        plt.legend(loc="best")
        plt.savefig("visualize/corr/corr_predict_sequence_relu_{}_{}_model".format(batch_size, epoch))
        plt.close()

    if mode == "all":
        plt.title("Predicted with All Data batch: {} (sequence)".format(batch_size))
        plt.plot(test_value, c="g", label="raw")
        plt.plot(predicted_values, c="b", alpha=.4, label="predicted")
        plt.legend(loc="best")
        plt.savefig("visualize/all/all_predict_sequence_relu_{}_{}".format(batch_size, epoch))
        plt.close()

def post_process(predicted_value):
    result = np.maximum(0, predicted_value)

    print("Post Processing Finished")
    return result

def main():
    time_steps = 48
    # Corr
    features1 = 4
    # All
    features2 = 8

    epoch = 3000
    batch_size = 32
    learning_rate = 0.00001

    # 상관관계 기반분석
    trainX1, trainy1, testX1, testy1 = create_sequence(time_steps, features1, "corr")
    model1 = initiate_model(learning_rate, time_steps, features1)

    trained_result1 = train_model(model1, trainX1, trainy1, 0.3, batch_size, epoch)

    # pd.DataFrame.from_dict(trained_result1.history).plot()

    loss1 = model1.evaluate(testX1, testy1)
    print(f"Corr Test loss: {loss1}")

    predicted1 = model1.predict(testX1)

    predicted1 = post_process(predicted1)

    visualize(predicted1, testy1, trained_result1, "corr",  str(epoch), str(batch_size))

    # 전체 데이터 사용
    trainX2, trainy2, testX2, testy2 = create_sequence(time_steps, features2, "all")
    model2 = initiate_model(learning_rate, time_steps, features2)

    trained_result2 = train_model(model2, trainX2, trainy2, 0.3, batch_size, epoch)

    # pd.DataFrame.from_dict(trained_result2.history).plot()

    loss2 = model2.evaluate(testX2, testy2)
    print(f"All Test loss: {loss2}")

    predicted2 = model2.predict(testX2)

    predicted2 = post_process(predicted2)

    visualize(predicted2, testy2, trained_result2, "all", str(epoch), str(batch_size))

main()
