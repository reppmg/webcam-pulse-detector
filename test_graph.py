import matplotlib.pyplot as plt
import pandas as pd

from transforms import transform

data = pd.read_csv("Webcam-pulse-k3_avi-dlib-dots.csv", sep=",").to_numpy()
time, bpm = data[250:, 0], data[250:, 1]
set = "P1LC4"
# time_base, bpm_base = read_file("/Users/maksimrepp/Documents/nir/public_sheet/%s/%s_Mobi_RR-intervals.bpm" % (set, set))
plt.figure()
time, bpm, alg = transform(time, bpm)
plt.plot(time, bpm)
plt.show()
plt.savefig("%s.png" % set)
#
# def shit():
#     return "a", "b"
#
#
# a, b = shit()
# print(a)
