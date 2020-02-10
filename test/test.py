import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

fig, ax = plt.subplots(1, 1)

while True:
    # If wanting to see an "animation" of points added, add a pause to allow the plotting to take place
    x = range(10)
    y = np.random.rand(10)
    ax.scatter(x, y)