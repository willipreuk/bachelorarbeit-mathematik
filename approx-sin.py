import numpy as np
import pandas as pd
from keras import layers
import keras
import matplotlib.pyplot as plt


def fun(x_vals):
    return np.cos(np.pi + 5 * x_vals) + np.sin(2 * x_vals + 2)


x = np.linspace(0, 2 * np.pi, 10000)
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
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.MeanSquaredError,
)

print(np.shape(train_features["x"]))

history = model.fit(
    train_features["x"],
    train_labels,
    batch_size=64,
    epochs=20,
    validation_split=0.2,
)

x_plot = test_features["x"]
y = model.predict(x_plot)

plt.plot(x_plot, y, label='Predictions')
plt.plot(x_plot, test_labels, label='True values')
plt.legend()
plt.show()
