from dbm import error

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from data import read_data
from keras import layers, Input, Sequential, callbacks, models, optimizers, ops, losses, saving
from sklearn.model_selection import train_test_split
import os

from simulation.track.data import read_test_data, test_function

error_weight = 0.75
first_diff_weight = 0.5


def custom_loss(y_true, y_pred):
    error = losses.mean_absolute_error(y_true, y_pred)
    first_diff = ops.mean(ops.square(ops.diff(y_pred, axis=0)))
    second_diff = ops.mean(ops.square(ops.diff(y_pred, n=2, axis=0)))

    return (error_weight * error
            + (1 - error_weight) * first_diff_weight * first_diff
            + (1 - error_weight) * (1 - first_diff_weight) * second_diff)


def train_nn():
    model_path = "keras-models/TEST2_252_2_relu_error1_diff0.model.keras"
    data, x_vals = read_data()

    print("Data shape: ", data.shape)
    print("x_vals shape: ", x_vals.shape)

    x_train = x_vals
    y_train = data
    x_valid = x_vals
    y_valid = data

    feature_normalizer = layers.Normalization(axis=None)
    feature_normalizer.adapt(x_train)

    if os.path.exists(model_path):
        model = models.load_model(model_path, custom_objects={'custom_loss': custom_loss})
        print("Model loaded")
    else:
        model = Sequential([
            Input(shape=(1,)),
            feature_normalizer,
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation="linear")
        ])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=custom_loss,
        )

    print(model.summary())

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='loss',
        save_best_only=True,
        save_freq=100
    )

    history = model.fit(
        x_train,
        y_train,
        # batch size must be min 3 for diff to work
        batch_size=16,
        epochs=3000,
        initial_epoch=0,
        validation_data=(x_valid, y_valid),
        # callbacks=[model_checkpoint_callback]
    )

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    nn_x = np.arange(0, 10, 0.0001)

    plt.figure()

    # model data
    plt.plot(nn_x, model.predict(nn_x), label="NN")

    # filtered spline
    fft_result = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), d=x_vals[1] - x_vals[0])
    cutoff_freq = 0.75
    fft_result[np.abs(fft_freq) > cutoff_freq] = 0
    filtered_signal = np.fft.ifft(fft_result)
    plt.plot(nn_x, sp.interpolate.CubicSpline(x_vals, filtered_signal)(nn_x), label="Filtered spline")

    # real data
    # plt.plot(nn_x, test_function(nn_x), 'r')

    plt.plot(x_vals, data, 'x', label="data")
    # plt.plot(x_valid, y_valid, 'bx')

    plt.title('Model vs Data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_nn()
