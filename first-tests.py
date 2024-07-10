import numpy as np
from ucimlrepo import fetch_ucirepo
import keras
from keras import layers
import tensorflow as tf
from matplotlib import pyplot as plt

air_quality = fetch_ucirepo(id=360)
print("Dataset fetched")

dataset = air_quality.data.features

# Missing values are tagged with -200 value.
dataset.replace(-200, np.nan, inplace=True)

dataset.drop("Date", axis=1, inplace=True)
dataset.drop("Time", axis=1, inplace=True)

feature = "C6H6(GT)"

print("Shape of datapoints: ", np.shape(dataset))

clean_dataset = dataset[[feature]].dropna()

train_dataset = clean_dataset.sample(frac=0.8, random_state=0)
test_dataset = clean_dataset.drop(train_dataset.index)

print("Shape of training datapoints: ", np.shape(train_dataset))
print("Shape of testing datapoints: ", np.shape(test_dataset))

feature_normalizer = layers.Normalization(axis=-1)
feature_normalizer.adapt(np.array(train_dataset))

model = keras.Sequential(
    [
        feature_normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.mean_absolute_error,
)

print(model.summary())

x_train = np.array(train_dataset.index).reshape((len(train_dataset.index), 1))
y_train = np.array(train_dataset)
x_test = np.array(test_dataset.index).reshape((len(test_dataset.index), 1))
y_test = np.array(test_dataset)

print(np.shape(np.array(train_dataset.index).reshape((len(train_dataset.index), 1))))
print(np.shape(train_dataset))

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=100,
    validation_data=(x_test, y_test),
    verbose=1
)

x = tf.linspace(0.0, 8000, 1000)
y = model.predict(x)

plt.plot(x, y, label='Predictions')
plt.show()


