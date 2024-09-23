from enum import Enum


class Excitations(Enum):
    SIMULATED = 1
    DATA_SPLINE = 2
    DATA_NEURAL_NETWORK = 3


class NeuralNetworkConfig:
    batch_size = 128
    epochs = 1000

    # error_weight = 0.5
    # first_diff_weight = 0.5

    # error_weight = 0.75
    # first_diff_weight = 1
    # model_path = "keras-models/data_1.model.keras"

    # error_weight = 1
    # first_diff_weight = 0
    # model_path = "keras-models/data_2.model.keras"

    # error_weight = 0.9
    # first_diff_weight = 1
    # model_path = "keras-models/data_3.model.keras"

    # error_weight = 0.5
    # first_diff_weight = 1
    # model_path = "keras-models/data_4.model.keras"

    # error_weight = 0.5
    # first_diff_weight = 0.5
    # model_path = "keras-models/data_5.model.keras"

    # error_weight = 0.75, first_diff_weight = 1
    # model_path = "keras-models/simulated_1.model.keras"

    error_weight = 0.9
    first_diff_weight = 1
    model_path = "keras-models/simulated_2.model.keras"

    # error_weight = 0.9, first_diff_weight = 1
    # model_path = "keras-models/simulated_3.model.keras"

    # error_weight = 0.5, first_diff_weight = 1
    # model_path = "keras-models/simulated_4.model.keras"


class Config:
    excitation = Excitations.DATA_NEURAL_NETWORK

    t_end = 10
    delta_t = 0.025
    # delta_t = 0.089993

    data_r_path = "data/Hhwayre.dat"
    data_l_path = "data/Hhwayli.dat"
