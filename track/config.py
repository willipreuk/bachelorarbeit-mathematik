from enum import Enum


class Excitations(Enum):
    SIMULATED = 1
    DATA_SPLINE = 2
    DATA_NEURAL_NETWORK = 3


class NeuralNetworkConfig:
    error_weight = 1
    first_diff_weight = 0

    model_path = "keras-models/TEST2_252_2_relu_error1_diff0.model.keras"

    epochs = 1000


class Config:
    excitation = Excitations.DATA_NEURAL_NETWORK

    t_end = 10
    delta_t = 0.089993

    data_r_path = "data/Hhwayre.dat"
    data_l_path = "data/Hhwayli.dat"