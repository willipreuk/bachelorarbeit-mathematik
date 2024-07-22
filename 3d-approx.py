import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import layers
import keras


def calculate_z(x, y):
    return np.e ** x * np.cos(y) * np.sin(x + np.pi)


x_data = np.linspace(0, np.pi, 1000)
y_data = np.linspace(0, np.pi, 1000)
x_data_mesh, y_data_mesh = np.meshgrid(x_data, y_data)
x_data_mesh_flat = x_data_mesh.flatten()
y_data_mesh_flat = y_data_mesh.flatten()

z_data = calculate_z(x_data_mesh_flat, y_data_mesh_flat)

dataset = pd.DataFrame({"x": x_data_mesh_flat, "y": y_data_mesh_flat, "z": z_data})

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('z')
test_labels = test_features.pop('z')

print("Shape of training datapoints: ", np.shape(train_dataset))
print("Shape of testing datapoints: ", np.shape(test_dataset))

feature_normalizer = layers.Normalization(axis=-1)
feature_normalizer.adapt(np.array(train_features))

print(feature_normalizer.mean.numpy())

model = keras.Sequential(
    [
        feature_normalizer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.mean_absolute_error,
)

model.fit(
    train_features,
    train_labels,
    batch_size=128,
    epochs=10,
    validation_data=(test_features, test_labels)
)

fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.linspace(0, np.pi, 100)
y = np.linspace(0, np.pi, 100)
X, Y = np.meshgrid(x, y)

x_flat = X.flatten()
y_flat = Y.flatten()
predict_dataset = pd.DataFrame({"x": x_flat, "y": y_flat})
z_pred = model.predict(predict_dataset)
Z_pred = z_pred.reshape(X.shape)
ax.plot_surface(X, Y, Z_pred, rstride=1, cstride=1, cmap='inferno', edgecolor='none')

Z = calculate_z(X, Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

a = np.linspace(0, np.pi, 100)
Z_line = calculate_z(a, a)
#ax.plot3D(a, a, Z_line, 'red')

fig.show()
plt.show()
