import numpy as np
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [0, 1.2, 2.3, 1, 0.4, -0.5, -1.2, -2, -0.9, 0]

figure = plt.figure()
ax = figure.add_subplot(111)
ax.plot(x, y)
ax.set_xlabel("time")
ax.set_ylabel("x acceleration")
plt.show()
