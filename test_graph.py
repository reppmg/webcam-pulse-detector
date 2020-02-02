import matplotlib.pyplot as plt
import pandas as pd


def read_file(file):
    try:
        lines = open(file, 'r').readlines()
    except:
        lines = open(file.replace("_e", "-e"))
    time, bpm = [], []
    for i in lines:
        splt = i.split(" ")
        time.append(float(splt[0]))
        bpm.append(float(splt[1]))
    start_time = time[0]
    time = [i - start_time for i in time]
    return time, bpm


# data = pd.read_csv("Webcam-pulse-v_cl2.csv", sep=",").to_numpy()
# time, bpm = data[:, 0], data[:, 1]
# set = "P1LC4"
# # time_base, bpm_base = read_file("/Users/maksimrepp/Documents/nir/public_sheet/%s/%s_Mobi_RR-intervals.bpm" % (set, set))
# plt.figure()
# plt.plot(time, bpm)
# plt.show()
# plt.savefig("%s.png" % set)
#
# def shit():
#     return "a", "b"
#
#
# a, b = shit()
# print(a)
