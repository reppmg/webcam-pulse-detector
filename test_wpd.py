import os
import runpy
import sys
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from test_graph import read_file
from experiments import transform

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
            # bashCommand = '%s %s -v %s' % (path_to_python, path_to_script, path_to_video)
            # print(bashCommand)
            #
            # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            # output, error = process.communicate()
            # print(output)

            folder = root.split("/")[-1]
            out_file = "Webcam-pulse-%s.csv" % folder.lower()
            data = pd.read_csv(out_file).to_numpy()
            time, bpm = data[:, 0], data[:, 1]
            time, bpm = transform(time, bpm)
            time_base, bpm_base = read_file(
                "/Users/maksimrepp/Documents/nir/public_sheet/%s/%s_Mobi_RR-intervals.bpm" % (folder, folder))
            plt.figure()
            plt.plot(time, bpm, time_base, bpm_base)
            plt.title(folder)
            plt.savefig(
                "/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/savinsky/%s.png" % folder)
            plt.close()

