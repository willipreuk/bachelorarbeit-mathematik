import matplotlib.pyplot as plt
import numpy as np

from data import read_data
from keras import layers, Input, Sequential, callbacks, models, optimizers, losses
from sklearn.model_selection import train_test_split
import os


def train_nn():
    model_path = "keras-models/512_2_relu.model.keras"
    data, x_vals = read_data()

    x_train, x_valid, y_train, y_valid = train_test_split(x_vals, data, test_size=0.33, shuffle=True)

    feature_normalizer = layers.Normalization(axis=None)
    feature_normalizer.adapt(x_train)

    if os.path.exists(model_path):
        model = models.load_model(model_path)
        print("Model loaded")
    else:
        model = Sequential([
            Input(shape=(1,)),
            feature_normalizer,
            layers.Dense(512, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation="linear")
        ])

        model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.mean_squared_error,
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
        batch_size=2,
        epochs=1,
        initial_epoch=0,
        validation_data=(x_valid, y_valid),
        callbacks=[model_checkpoint_callback]
    )

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    nn_x = np.arange(0, 10, 0.005)
    predicted = model.predict(nn_x)

    plt.figure()
    plt.plot(nn_x, model.predict(nn_x))
    plt.plot(x_vals, data, 'x')
    plt.show()


if __name__ == '__main__':
    train_nn()
