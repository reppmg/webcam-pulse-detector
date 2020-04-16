import os
import runpy
import numpy as np
import sklearn.metrics
import sys
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from read_files import read_file
from experiments import trim_real
from transforms import transform

import shutil

path_to_script = '/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/get_pulse.py'
path_to_python = '/usr/local/bin/python3.7'

walk = os.walk("/Users/maksimrepp/Documents/nir/public_sheet")
next(walk)
for root, dirs, files in walk:
    for file in files:
        if file.endswith(".mp4"):
            filename = file.split(".")[0]
            path_to_video = os.path.join(root, file)
            sys.argv[0] = '-v %s' % path_to_video
            bashCommand = '%s %s -v %s' % (path_to_python, path_to_script, path_to_video)
            print(bashCommand)

            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)

            folder = root.split("/")[-1]
            out_file = "results_raw/Webcam-pulse-%s-dlib-dots.csv" % filename
            backup_file = "results_raw_cutlow6/Webcam-pulse-%s-dlib-dots.csv" % filename

            shutil.copyfile(out_file, backup_file)

            data = pd.read_csv(out_file).to_numpy()
            time, bpm = data[250:, 0], data[250:, 1]
            time -= time[0]
            ot = time.copy()
            op = bpm.copy()
            # time, bpm, alg = transform(time, bpm)
            time_real, bpm_real = read_file(
                "/Users/maksimrepp/Documents/nir/public_sheet/%s/%s_Mobi_RR-intervals.bpm" % (folder, folder))
            time_real = time_real[0:-20]
            bpm_real = bpm_real[0:-20]

            time_real_trim, bpm_real_trim = trim_real(time_real, time, bpm_real)
            bpm_est_interp = np.interp(time_real_trim, time, bpm)

            error = sklearn.metrics.mean_squared_error(bpm_real_trim, bpm_est_interp)

            plt.figure()
            plt.plot(time_real_trim, bpm_est_interp, time_real_trim, bpm_real_trim)
            plt.title("%s. MSE: %.2f" % (folder, error))
            plt.legend(["estimated", "real"])
            plt.savefig(
                "/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/results_raw_cutlow6/%s.png" % folder)
            plt.close()
