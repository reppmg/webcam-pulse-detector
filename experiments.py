import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from test_graph import read_file

bpm_file = "/Users/maksimrepp/Documents/nir/public_sheet/P1LC5/P1LC5_Mobi_RR-intervals.bpm"
csv_file = "/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/Webcam-pulse-P1LC5.csv"


def transform(times, pulse) :
    return times, pulse


data = pd.read_csv(csv_file).to_numpy()
times = data[:, 0]
pulse = data[:, 1]
times, pulse = transform(times, pulse)
expected_times, expected_pulse = read_file(bpm_file)
plt.figure()
plt.title("No changes")
plt.plot(times, pulse, expected_times, expected_pulse)
plt.show()
