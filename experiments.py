import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from test_graph import read_file
from scipy.signal import savgol_filter
from smooth import smooth

np.random.seed(1)
LC = "1"
L = "1"
folder = "P%sH%s" % (L, LC)
# folder = "P%sLC%s" % (L, LC)
bpm_file = "/Users/maksimrepp/Documents/nir/public_sheet/%s/%s_Mobi_RR-intervals.bpm" % (folder, folder)
csv_file = "/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/Webcam-pulse-%s.csv" % (folder)


def transform(times, pulse):
    percentile = np.percentile(pulse, 90)
    percentile_low = np.percentile(pulse, 15)
    pulse = np.where(pulse < percentile, pulse, percentile)
    pulse = np.where(pulse > percentile_low, pulse, percentile_low)
    pulse = savgol_filter(pulse, 51, 3)
    return times, pulse[0:len(times)]


data = pd.read_csv(csv_file).to_numpy()
times = data[:, 0]
pulse = data[:, 1]
times, pulse = transform(times, pulse)
expected_times, expected_pulse = read_file(bpm_file)
plt.figure()
title = "Savinsky + percentile window"
plt.title(title)
plt.plot(times, pulse, expected_times, expected_pulse)
plt.legend(["estimated", "real"])
plt.show()
# plt.savefig("experiments/%s.png" % title)
