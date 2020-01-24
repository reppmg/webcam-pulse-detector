import matplotlib.pyplot as plt


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


# time, bpm = read_file("bpm_p1lc4_edited.txt")
# set = "P1LC4"
# time_base, bpm_base = read_file("/Users/maksimrepp/Documents/nir/public_sheet/%s/%s_Mobi_RR-intervals.bpm" % (set, set))
# plt.figure()
# plt.plot(time, bpm, time_base, bpm_base)
# plt.show()
# plt.savefig("%s.png" % set)
