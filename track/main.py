import matplotlib.pyplot as plt
import numpy as np

from data import read_data
from keras import layers, Input, Sequential, callbacks, models, optimizers, ops, losses, saving
from sklearn.model_selection import train_test_split
import os


@saving.register_keras_serializable()
class CustomLoss():
    def __init__(self, error_weight, first_diff_weight, name="custom_loss", **kwargs):
        self.error_weight = error_weight
        self.first_diff_weight = first_diff_weight * (1 - error_weight)
        self.second_diff_weight = (1 - first_diff_weight) * (1 - error_weight)

    def __call__(self, y_true, y_pred):
        error = ops.mean(ops.abs(y_true - y_pred), axis=-1)
        first_diff = ops.mean(ops.square(ops.diff(y_pred, axis=0)))
        second_diff = ops.mean(ops.square(ops.diff(y_pred, n=2, axis=0)))

        return self.error_weight * error + self.first_diff_weight * first_diff + self.second_diff_weight * second_diff


def train_nn():
    model_path = "keras-models/512_2_relu.model.keras"
    data, x_vals = read_data()

    x_train, x_valid, y_train, y_valid = train_test_split(x_vals, data, test_size=0.5, shuffle=True)

    feature_normalizer = layers.Normalization(axis=None)
    feature_normalizer.adapt(x_train)

    if os.path.exists(model_path):
        model = models.load_model(model_path)
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
            loss=CustomLoss(0.2, 0.75),
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
        batch_size=32,
        epochs=15000,
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
    plt.plot(nn_x, model.predict(nn_x))
    plt.plot(x_vals, data, 'x')
    plt.show()


if __name__ == '__main__':
    train_nn()
