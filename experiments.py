import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from read_files import read_file
import sklearn.metrics
from transforms import transform

# np.random.seed(1)
# LC = "1"
# L = "1"
# folder = "P%sH%s" % (L, LC)
# # folder = "P%sLC%s" % (L, LC)
# bpm_file = "/Users/maksimrepp/Documents/nir/public_sheet/%s/%s_Mobi_RR-intervals.bpm" % (folder, folder)
# file = "k1-dlib-dots"
# csv_file = "/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/Webcam-pulse-%s.csv" % (file)


def trim_real(time_real, time, bpm_real):
    trim_index = np.where(time_real > time[0])[0]
    time_real_trim = np.take(time_real, trim_index)
    bpm_real_trim = np.take(bpm_real, trim_index)
    return time_real_trim, bpm_real_trim

#
# data = pd.read_csv(csv_file).to_numpy()
# times = data[250:, 0]
# times = times - times[0]
# pulse = data[250:, 1]
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
# plt.savefig("experiments/%s-%s.png" % (file, title))
