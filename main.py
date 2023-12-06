from lstm import initiate_model
from preprocess import create_sequence
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

def train_model(model, trainX, trainy, validation_split: int, batch_size: int, epoch: int):
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=5, mode="auto")

    trained = model.fit(
        trainX,
        trainy,
        validation_split=validation_split, 
        batch_size=batch_size,
        epochs=epoch,
        callbacks = [early_stopping]
    )

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

    plt.savefig("visualize/history/scaled_train_history_{}_e{}_b{}".format(mode, epoch, batch_size))
    plt.close()

    if mode == "corr":
        plt.title("Predicted with Correlated Data batch: {} (sequences)".format(batch_size))
        plt.plot(predicted_values, c="b", label="predicted")
        plt.plot(test_value, c="g", alpha=.5, label="raw")
        plt.legend(loc="best")
        plt.savefig("visualize/corr/scaled_corr_predict_sequence_relu_{}_{}_model".format(batch_size, epoch))
        plt.close()

    if mode == "all":
        plt.title("Predicted with All Data batch: {} (sequence)".format(batch_size))
        plt.plot(test_value, c="g", label="raw")
        plt.plot(predicted_values, c="b", alpha=.4, label="predicted")
        plt.legend(loc="best")
        plt.savefig("visualize/all/scaled_all_predict_sequence_relu_{}_{}".format(batch_size, epoch))
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
    batch_size = 64
    learning_rate = 0.0001

    # 상관관계 기반분석
    trainX1, trainy1, testX1, testy1 = create_sequence(time_steps, features1, "corr")
    model1 = initiate_model(learning_rate, time_steps, features1)

    trained_result1 = train_model(model1, trainX1, trainy1, 0.2, batch_size, epoch)

    loss1 = model1.evaluate(testX1, testy1)
    print(f"Corr Test loss: {loss1}")

    predicted1 = model1.predict(testX1)
    model1.save("model_result/lstm_model_corr_scaled_b{}_e{}_f{}".format(batch_size, epoch, features1), save_format="h5")

    post_predicted1 = post_process(predicted1)

    visualize(post_predicted1, testy1, trained_result1, "corr",  str(epoch), str(batch_size))
    mse1 = mean_squared_error(testy1, post_predicted1)
    rmse1 = np.sqrt(mse1)

    # 전체 데이터 사용
    trainX2, trainy2, testX2, testy2 = create_sequence(time_steps, features2, "all")
    model2 = initiate_model(learning_rate, time_steps, features2)

    trained_result2 = train_model(model2, trainX2, trainy2, 0.2, batch_size, epoch)

    loss2 = model2.evaluate(testX2, testy2)
    print(f"All Test loss: {loss2}")

    predicted2 = model2.predict(testX2)

    post_predicted2 = post_process(predicted2)
    model2.save("model_result/lstm_model_all_scaled_b{}_e{}_f{}".format(batch_size, epoch, features2), save_format="h5")

    visualize(post_predicted2, testy2, trained_result2, "all", str(epoch), str(batch_size))
    mse2 = mean_squared_error(testy2, post_predicted2)
    rmse2 = np.sqrt(mse2)

    eval_frame = pd.DataFrame(
        columns=["mse", "rmse"],
        index=["With Corr", "Without Corr"],
        data=[[round(mse1, 3), round(rmse1, 3)], [round(mse2, 3), round(rmse2)]]
    )

    eval_frame.to_csv("eval/scaled_evaluation_b{}_e{}_lr{}".format(str(batch_size), str(epoch), str(learning_rate)))
    print("finished. There are Evaluate Stats: {}".format(eval_frame))
main()
