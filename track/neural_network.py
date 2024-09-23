from matplotlib import pyplot as plt
from data import read_data
from keras import layers, Input, Sequential, callbacks, models, optimizers, ops, losses, saving
from config import NeuralNetworkConfig
import tensorflow as tf
import os

tf.random.set_seed(21)


def _custom_loss(y_true, y_pred):
    error = losses.mean_absolute_error(y_true, y_pred)
    first_diff = ops.mean(ops.square(ops.diff(y_pred, axis=0)))
    second_diff = ops.mean(ops.square(ops.diff(y_pred, n=2, axis=0)))

    return (NeuralNetworkConfig.error_weight * error
            + (1 - NeuralNetworkConfig.error_weight) * NeuralNetworkConfig.first_diff_weight * first_diff
            + (1 - NeuralNetworkConfig.error_weight) * (1 - NeuralNetworkConfig.first_diff_weight) * second_diff)


def _load_model():
    model_path = NeuralNetworkConfig.model_path

    if os.path.exists(model_path):
        model = models.load_model(model_path, custom_objects={'_custom_loss': _custom_loss})
        print("Model loaded")
    else:
        # use only the left side for normalizing, mean and variance should be the same
        _, _, x_vals = read_data()

        feature_normalizer = layers.Normalization(axis=None)
        feature_normalizer.adapt(x_vals)

        model = Sequential([
            Input(shape=(1,)),
            feature_normalizer,
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation="linear")
        ])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=_custom_loss,
        )

    print(model.summary())

    return model


# only load the model once
_model = _load_model()


def train_nn():
    data, data_l, x_vals = read_data()

    print("Data shape: ", data.shape)
    print("x_vals shape: ", x_vals.shape)

    x_train = x_vals
    y_train = data
    x_valid = x_vals
    y_valid = data

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=NeuralNetworkConfig.model_path,
        monitor='loss',
        save_best_only=True,
        save_freq=100
    )

    history = _model.fit(
        x_train,
        y_train,
        # batch size must be min 3 for diff to work
        batch_size=NeuralNetworkConfig.batch_size,
        epochs=NeuralNetworkConfig.epochs,
        validation_data=(x_valid, y_valid),
        callbacks=[model_checkpoint_callback]
    )

    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model loss')
    plt.legend()


def predict(t):
    return _model.predict(t, verbose=0)
