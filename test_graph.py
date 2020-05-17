import matplotlib.pyplot as plt
import pandas as pd

from transforms import transform

video = "k4"
data = pd.read_csv("results_raw/Webcam-pulse-cam-dlib-dots.csv", sep=",").to_numpy()
time, bpm = data[250:, 0], data[250:, 1]

# time_base, bpm_base = read_file("/Users/maksimrepp/Documents/nir/public_sheet/%s/%s_Mobi_RR-intervals.bpm" % (set, set))
time, bpm, alg = transform(time, bpm)
plt.plot(time, bpm)
# plt.show()
plt.savefig("%s.png" % video)
#
# def shit():
#     return "a", "b"
#
#
# a, b = shit()
# print(a)
