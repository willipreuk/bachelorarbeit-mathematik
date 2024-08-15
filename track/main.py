import matplotlib.pyplot as plt
from data import read_data

if __name__ == '__main__':
    data, x_vals = read_data()

    plt.plot(x_vals, data, 'x')
    plt.show()
