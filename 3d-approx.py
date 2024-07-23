import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from keras import layers
import keras


def gamma(t):
    return [t, t ** 2]


def gamma_dot(t):
    return np.sqrt(1 + 2 * t ** 2)


def calculate_z(x, y):
    return np.e ** x * np.cos(y) * np.sin(x + np.pi)


def weg(t):
    x, y = gamma(t)
    return calculate_z(x, y) * gamma_dot(t)


x_data = np.linspace(0, np.pi, 100)
y_data = np.linspace(0, np.pi, 100)
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

model = keras.Sequential(
    [
        feature_normalizer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation="linear")
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.mean_absolute_error,
)

history = model.fit(
    train_features,
    train_labels,
    batch_size=128,
    epochs=200,
    validation_data=(test_features, test_labels)
)


def weg_integral():
    t_start = 0
    t_end = 1

    t = np.linspace(t_start, t_end, 100)
    x, y = gamma(t)
    predict_dataset = pd.DataFrame({"x": x, "y": y})
    z = model.predict(predict_dataset)
    integral_pred = sp.integrate.trapezoid(z.flatten() * gamma_dot(t), t)
    print("Predicted integral: ", integral_pred)
    integral_correct = sp.integrate.quad(weg, t_start, t_end)
    print("Correct integral: ", integral_correct[0])


def print_model():
    x = np.linspace(0, np.pi, 100)
    y = np.linspace(0, np.pi, 100)
    X, Y = np.meshgrid(x, y)

    x_flat = X.flatten()
    y_flat = Y.flatten()
    predict_dataset = pd.DataFrame({"x": x_flat, "y": y_flat})
    z_pred = model.predict(predict_dataset)
    Z_pred = z_pred.reshape(X.shape)

    Z = calculate_z(X, Y)

    # First figure: 3D surface plot
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, Z_pred, rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none') # Display the first figure

    # Second figure: Model loss over epochs
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='test')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper left')
    plt.show()  # Display the second figure


weg_integral()
