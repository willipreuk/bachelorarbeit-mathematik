from enum import Enum


class TrainData(Enum):
    SIMULATED = "simulated"
    DATA = "data"


class Excitations(Enum):
    SIMULATED = "simulated excitations"
    SIMULATED_NEURAL_NETWORK = "simulated neural network excitations"
    DATA_SPLINE = "data spline excitations"
    DATA_NEURAL_NETWORK = "data neural network excitations"
    DATA_NEURAL_NETWORK_PREDICT = "data neural network excitations prediction"
    SIMULATED_NEURAL_NETWORK_PREDICT = "simulated neural network excitations prediction"



first_diff_weight = 0
second_diff_weight = 0
def get_model_path():
    return "keras-models/" + str(first_diff_weight) + "_" + str(second_diff_weight) + "_" + data_source.value + ".model.keras"

excitation = Excitations.SIMULATED_NEURAL_NETWORK
data_source = TrainData.SIMULATED

t_end = 3
delta_t = 0.025
delta_t_simulation = delta_t / 10

batch_size = 400
epochs = 750

data_r_path = "data/Hhwayre.dat"
data_l_path = "data/Hhwayli.dat"
phase_shift = 0.4

neural_network_predict_delta_t = delta_t
