import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from test_graph import read_file
from scipy.signal import savgol_filter
from smooth import smooth

np.random.seed(1)
bpm_file = "/Users/maksimrepp/Documents/nir/public_sheet/P1LC5/P1LC5_Mobi_RR-intervals.bpm"
csv_file = "/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/Webcam-pulse-P1LC5.csv"


def transform(times, pulse):
    pulse = smooth(pulse, 11, 'flat')
    return times, pulse[0:len(times)]


data = pd.read_csv(csv_file).to_numpy()
times = data[:, 0]
pulse = data[:, 1]
times, pulse = transform(times, pulse)
expected_times, expected_pulse = read_file(bpm_file)
plt.figure()
title = "Flat window"
plt.title(title)
plt.plot(times, pulse, expected_times, expected_pulse)
# plt.show()
plt.savefig("experiments/%s.png" % title)
