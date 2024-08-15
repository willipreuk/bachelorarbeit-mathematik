from xml.sax.handler import feature_external_pes
from sklearn.model_selection import train_test_split

import keras
from keras import layers, Input
import numpy as np
from numpy import ndarray


class NeuralNetwork:
    def __init__(self, features: ndarray, labels: ndarray):
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.33, shuffle=True)

        self.train_features = x_train
        self.train_labels = y_train
        self.validation_features = x_valid
        self.validation_labels = y_valid

        self.model = self.create_model()

    def train(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.huber,
        )

        self.model.fit(
            self.train_features,
            self.train_labels,
            batch_size=32,
            epochs=1000,
            validation_data=(self.validation_features, self.validation_labels),
        )

    def predict(self, test_features: ndarray) -> ndarray:
        return self.model.predict(test_features)

    def create_model(self) -> keras.Sequential:
        feature_normalizer = layers.Normalization(axis=None)
        feature_normalizer.adapt(self.train_features)

        return keras.Sequential([
            Input(shape=(1,)),
            feature_normalizer,
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation="linear")
        ])
