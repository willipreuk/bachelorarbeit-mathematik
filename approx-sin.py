import numpy as np
import pandas as pd
from keras import layers
import keras
import matplotlib.pyplot as plt
import scipy as sp


def fun(x_vals):
    # runge's function
    # return 1/(1 + 25 * x_vals**2)

    return np.heaviside(x, 0.5) * np.cos(np.pi + 5 * x_vals) + np.sin(2 * x_vals + 2)


x = np.linspace(-2, 2, 10000)
y = fun(x)

dataset = pd.DataFrame({"x": x, "y": y})

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('y')
test_labels = test_features.pop('y')

print("Shape of training datapoints: ", np.shape(train_dataset))
print("Shape of testing datapoints: ", np.shape(test_dataset))

feature_normalizer = layers.Normalization(axis=-1)
feature_normalizer.adapt(np.array(train_features))

print(feature_normalizer.mean.numpy())

model = keras.Sequential(
    [
        feature_normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.mean_squared_error,
)

model.fit(
    train_features["x"],
    train_labels,
    batch_size=64,
    epochs=20,
    validation_split=0.1,
)

x_plot = test_features["x"]
y = model.predict(x_plot)

interpolate_dataset = train_dataset.copy()
interpolate_dataset.sort_values(by="x", inplace=True)
y_interp = sp.interpolate.CubicSpline(interpolate_dataset["x"].to_numpy(), interpolate_dataset.pop("y").to_numpy())

plt.plot(x_plot, y, label='Predictions - MSE')
plt.plot(x_plot, y_interp(x_plot), label='Interpolation')
# plt.plot(x_plot, y_interp(x_plot, 2), label='Interpolation - second derivative')
plt.plot(x_plot, test_labels, 'g:', label='Actual Data')

plt.legend()
plt.show()
