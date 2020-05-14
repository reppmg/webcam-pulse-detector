import os
import runpy
import numpy as np
import sklearn.metrics
import sys
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from read_files import *
from experiments import trim_real
from transforms import transform

path_to_script = '/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/get_pulse.py'
path_to_python = '/usr/local/bin/python3.7'
filename = "k_5"
path_to_video = "/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/k_videos/%s.mp4" % filename

#
# sys.argv[0] = '-v %s' % path_to_video
# bashCommand = '%s %s -v %s' % (path_to_python, path_to_script, path_to_video)
# print(bashCommand)
#
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()
# print(os.getcwd())
# print(output)
#


out_file = "../results_raw/Webcam-pulse-%s-dlib-dots.csv" % filename
data = pd.read_csv(out_file).to_numpy()
time, bpm = data[0:, 0], data[0:, 1]
time -= time[0]
ot = time.copy()
op = bpm.copy()
# time, bpm, alg = transform(time, bpm)
time_real, bpm_real = read_file_mio("%s.csv" % filename)
time_real = time_real[0:]
bpm_real = bpm_real[0:]

time_real_trim, bpm_real_trim = trim_real(time_real, time, bpm_real)
bpm_est_interp = np.interp(time_real_trim, time, bpm)

error = sklearn.metrics.mean_squared_error(bpm_real_trim, bpm_est_interp)

plt.figure()
plt.plot(time_real_trim, bpm_est_interp, time_real_trim, bpm_real_trim)
plt.title("%s. MSE: %.2f" %(filename, error))
plt.legend(["estimated", "real"])
plt.savefig(
    "/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/k_videos/%s.png" % filename)
plt.close()

