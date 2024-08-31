import os
from sklearn.model_selection import train_test_split
from keras import layers, Input, optimizers, losses, Sequential, callbacks, models
from numpy import ndarray


class NeuralNetwork:
    def __init__(self, features: ndarray, labels: ndarray):
        self.checkpoint_path = "models/sequential-interpolated-4096-500.model.keras"

        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.33, shuffle=True)

        self.scale = (1 - max(y_train)) / max(y_train)
        print("Scale: ", self.scale)
        self.train_features = x_train
        self.train_labels = y_train * self.scale
        self.validation_features = x_valid
        self.validation_labels = y_valid * self.scale

    def train(self):
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

        model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            monitor='loss',
            save_best_only=True
        )

        return self.model.fit(
            self.train_features,
            self.train_labels,
            batch_size=320,
            epochs=500,
            initial_epoch=0,
            validation_data=(self.validation_features, self.validation_labels),
            callbacks=[model_checkpoint_callback]
        )

    def predict(self, input_data: ndarray) -> ndarray:
        return self.model.predict(input_data) / self.scale

    def load(self):
        if os.path.exists(self.checkpoint_path):
            self.model = models.load_model(self.checkpoint_path)
            print("Model loaded")
            print(self.model.summary())
            return True
        else:
            return False

    def create_model(self):
        feature_normalizer = layers.Normalization(axis=None)
        feature_normalizer.adapt(self.train_features)

        model = Sequential([
            Input(shape=(1,)),
            feature_normalizer,
            layers.Dense(4096, activation='relu'),
            layers.Dense(4096, activation='relu'),
            layers.Dense(4096, activation='relu'),
            layers.Dense(4096, activation='relu'),
            layers.Dense(4096, activation='relu'),
            layers.Dense(1, activation="linear")
        ])

        model.compile(
            optimizer=optimizers.Adam(0.0001),
            loss=losses.mean_squared_error,
        )

        self.model = model
