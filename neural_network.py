import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from data import read_data
from keras import layers, Input, Sequential, callbacks, models, optimizers, ops, losses
import tensorflow as tf
import os
import config

tf.random.set_seed(21)


def _custom_loss(y_true, y_pred):
    error = losses.mean_squared_error(y_true, y_pred)
    first_diff = ops.mean(ops.square(ops.diff(y_pred, axis=0)))
    second_diff = ops.mean(ops.square(ops.diff(y_pred, n=2, axis=0)))

    return (error
            + config.first_diff_weigth * first_diff
            + config.second_diff_weigth * second_diff)


def _load_model():
    model_path = config.get_model_path()

    if os.path.exists(model_path):
        model = models.load_model(model_path, custom_objects={'_custom_loss': _custom_loss})
        print("Model loaded")
    else:
        # use only the left side for normalizing, mean and variance should be the same
        _, _, x_vals = read_data()

        mean = x_vals.mean()
        var = x_vals.var()
        print("Mean: ", mean)
        print("Variance: ", var)

        feature_normalizer = layers.Normalization(axis=None, mean=mean, variance=var)

        model = Sequential([
            Input(shape=(1,)),
            feature_normalizer,
            layers.Dense(128),
            layers.LeakyReLU(),
            layers.Dense(256),
            layers.LeakyReLU(),
            layers.Dense(512),
            layers.LeakyReLU(),
            layers.Dense(512),
            layers.LeakyReLU(),
            layers.Dense(1, activation='linear'),
        ])

        model.compile(
            optimizer=optimizers.Adam(1e-4),
            loss=_custom_loss,
        )

    print(model.summary())

    return model


def train_nn():
    model_path = config.get_model_path()

    model = _load_model()
    data, data_l, x_vals = read_data()

    x_vals_fine = np.arange(0, config.t_end, config.delta_t_simulation / 10)
    data_fine = np.interp(x_vals_fine, x_vals, data)
    x_vals = x_vals_fine
    data = data_fine

    print("Data shape: ", data.shape)
    print("x_vals shape: ", x_vals.shape)

    # not used
    x_train, x_valid, y_train, y_valid = train_test_split(x_vals, data, test_size=0.1, random_state=42)

    x_train = x_vals
    y_train = data
    x_valid = x_vals
    y_valid = data

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
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(x_valid, y_valid),
        callbacks=[model_checkpoint_callback]
    )

    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model loss')
    plt.legend()


def predict(t):
    model = _load_model()
    return model.predict(t, verbose=0)
