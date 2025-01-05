import config
from neural_network import train_nn


def train_alpha():
    config.second_diff_weight = 0
    config.first_diff_weight = 0
    train_nn()

    config.first_diff_weight = 0.01
    train_nn()

    config.first_diff_weight = 0.1
    train_nn()

    config.first_diff_weight = 0.25
    train_nn()


if __name__ == '__main__':
    train_nn()
