from enum import Enum


class TrainData(Enum):
    SIMULATED = "simulated"
    DATA = "data"


class Excitations(Enum):
    SIMULATED_SPLINE = "simulated excitations"
    SIMULATED_NEURAL_NETWORK = "simulated neural network excitations"
    DATA_SPLINE = "data spline excitations"
    DATA_NEURAL_NETWORK = "data neural network excitations"



first_diff_weigth = 0
second_diff_weigth = 0
def get_model_path():
    return "keras-models/" + str(first_diff_weigth) + "_" + str(second_diff_weigth) + "_" + data_source.value + ".model.keras"

excitation = Excitations.SIMULATED_NEURAL_NETWORK
data_source = TrainData.SIMULATED

t_end = 10
delta_t = 0.025
delta_t_simulation = delta_t / 10

batch_size = 400
epochs = 2000

data_r_path = "data/Hhwayre.dat"
data_l_path = "data/Hhwayli.dat"
