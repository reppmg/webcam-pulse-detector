import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constants import *
from test_graph import read_file
from scipy.signal import savgol_filter
import sklearn.metrics
from smooth import smooth

np.random.seed(1)
LC = "1"
L = "1"
folder = "P%sH%s" % (L, LC)
# folder = "P%sLC%s" % (L, LC)
bpm_file = "/Users/maksimrepp/Documents/nir/public_sheet/%s/%s_Mobi_RR-intervals.bpm" % (folder, folder)
csv_file = "/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/Webcam-pulse-%s.csv" % ("v_cl2")


def transform(times, pulse):
    if len(pulse) < 100:
        return times, pulse, "too short"
    percentile = np.percentile(pulse, 90)
    percentile_low = np.percentile(pulse, 10)
    mean_window_size = percentile_neighbours
    for i, p in enumerate(pulse):
        if p > percentile:
            start = max(0, i - mean_window_size)
            end = min(len(pulse), i + mean_window_size)
            pulse[i] = np.mean(pulse[start:end])
        if p < percentile_low:
            start = max(0, i - mean_window_size)
            end = min(len(pulse), i + mean_window_size)
            pulse[i] = np.mean(pulse[start:end])
    pulse = savgol_filter(pulse, savgol_window, savgol_power)
    return times, pulse[0:len(times)], "percMeanNeigh30 + savgol cutlow 2"


def trim_real(time_real, time, bpm_real):
    trim_index = np.where(time_real > time[0])[0]
    time_real_trim = np.take(time_real, trim_index)
    bpm_real_trim = np.take(bpm_real, trim_index)
    return time_real_trim, bpm_real_trim

#
# data = pd.read_csv(csv_file).to_numpy()
# times = data[:, 0]
# pulse = data[:, 1]
# ot = times.copy()
# op = pulse.copy()
# times, pulse, title = transform(times, pulse)
# expected_times, expected_pulse = read_file(bpm_file)
# expected_times, expected_pulse = trim_real(expected_times, times, expected_pulse)
# bpm_interp = np.interp(expected_times, times, pulse)
# error = sklearn.metrics.mean_squared_error(np.array(expected_pulse, dtype=np.double), np.array(bpm_interp, dtype=np.double))
# title = title + " MSE: %.2f" % error
# plt.figure()
# plt.title(title)
# plt.plot(times, pulse, expected_times, expected_pulse)
# plt.legend(["estimated", "real"])
# # plt.show()
# plt.savefig("experiments/%s.png" % title)
