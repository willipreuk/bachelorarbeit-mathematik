import numpy as np
import matplotlib.pyplot as plt
from model import calculate_z

x = np.linspace(0, 1/2*np.pi, 100)
y = np.linspace(0, 1/2*np.pi, 100)
X, Y = np.meshgrid(x, y)

Z = calculate_z(X, Y)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')  # Display the first figure

plt.show()