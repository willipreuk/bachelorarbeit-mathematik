from tensorflow.keras.optimizers import Adam
import keras
import numpy as np
import scipy as sp
from keras import layers
from keras import callbacks
from model import calculate_z, gamma, gamma_dot, weg
import pandas as pd

x_data = np.linspace(0, np.pi, 500)
y_data = np.linspace(0, np.pi, 500)
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


def create_model():
    feature_normalizer = layers.Normalization(axis=-1)
    feature_normalizer.adapt(np.array(train_features))

    return keras.Sequential([
        feature_normalizer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation="linear")
    ])


loss_functions = {
    'MeanSquaredError': keras.losses.mean_squared_error,
    'MeanAbsoluteError': keras.losses.mean_absolute_error,
    'Huber': keras.losses.huber
}

results = {}

# Loop through each loss function
for name, loss_fn in loss_functions.items():
    # Create a new instance of the model
    model = create_model()

    # Compile the model with the current loss function
    model.compile(optimizer=Adam(), loss=loss_fn, metrics=['mae'])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(train_features, train_labels, epochs=100, batch_size=32,
                        validation_data=(test_features, test_labels), verbose=1, callbacks=[early_stopping])

    # Evaluate the model on the test set
    test_loss, test_mae = model.evaluate(test_features, test_labels, verbose=0)

    t_start = 0
    t_end = 1
    t = np.linspace(t_start, t_end, 100)
    x, y = gamma(t)

    predict_dataset = pd.DataFrame({"x": x, "y": y})
    z_model = model.predict(predict_dataset, verbose=0)
    integral_pred = sp.integrate.trapezoid(z_model.flatten() * gamma_dot(t), t)

    z_calc = calculate_z(x, y)
    integral_calc = sp.integrate.trapezoid(z_calc * gamma_dot(t), t)

    error = integral_calc - integral_pred

    # Store the results
    results[name] = {
        'loss': test_loss,
        'mae': test_mae,
        'history': history.history,
        'error': error
    }

for name, result in results.items():
    print(f"Loss Function: {name}")
    print(f"Test Loss: {result['loss']}")
    print(f"Test MAE: {result['mae']}")
    print(f"Integral Error: {result['error']}")
    print("-" * 30)
