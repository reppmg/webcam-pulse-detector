import os
import argparse
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

parser = argparse.ArgumentParser(description='Webcam pulse detector.')
parser.add_argument('-f', default=None,
                    help='udp address:port destination for bpm data')
parser.add_argument("-g", default=0,
                    help="path to the (optional) video file")
args = parser.parse_args()
args = vars(args)

graph_arg = args["g"]
file_arg = args["f"]


def relative_error(bpm_est, bpm_real):
    relative_error = np.mean(np.abs(bpm_est - bpm_real) / bpm_real)
    # error = sklearn.metrics.mean_squared_error(bpm_real, bpm_est)
    return relative_error

path_to_script = '/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/get_pulse.py'
path_to_python = '/usr/local/bin/python3.7'

walk = os.walk("/Users/maksimrepp/Documents/nir/public_sheet")
next(walk)
for root, dirs, files in walk:
    for file in files:
        if file.endswith(".mp4"):
            if "P1H1" not in file:
                continue
            filename = "v_%s" % file_arg
            here = os.getcwd()
            path_to_video = os.path.join(here, filename+".mp4")
            sys.argv[0] = '-v %s' % path_to_video
            bashCommand = '%s %s -v %s' % (path_to_python, path_to_script, path_to_video)
            print(bashCommand)

            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)

            folder = root.split("/")[-1]
            suffix = "dlib-dots"
            out_file = "results_raw/Webcam-pulse-%s-%s.csv" % (filename, suffix)
            data = pd.read_csv(out_file).to_numpy()
            time, bpm = data[250:, 0], data[250:, 1]
            time -= time[0]

            ot = time.copy()
            op = bpm.copy()
            time, bpm, alg = transform(time, bpm)
            time_real, bpm_real = read_file(
                "/Users/maksimrepp/Documents/nir/public_sheet/%s/%s_Mobi_RR-intervals.bpm" % (folder, folder))
            time_real = time_real[0:-20]
            bpm_real = bpm_real[0:-20]

            time_real_trim, bpm_real_trim = trim_real(time_real, time, bpm_real)
            bpm_est_interp = np.interp(time_real_trim, time, bpm)

            error = sklearn.metrics.mean_squared_error(bpm_real_trim, bpm_est_interp)
            rel_error = relative_error(bpm_est_interp, bpm_real_trim)

            plt.figure()
            plt.plot(time_real_trim, bpm_est_interp, time_real_trim, bpm_real_trim)
            plt.title("%s. %s. MSE: %.2f    Relative error: %.2f%%" % (folder, graph_arg, error, rel_error * 100))
            plt.legend(["estimated", "real"])
            plt.savefig(
                "/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/spoil/%s.png" % filename)
            plt.close()
