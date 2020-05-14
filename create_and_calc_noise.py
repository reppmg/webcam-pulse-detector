import subprocess
import sys

vers = [
    ["noise_med", '"Noise-medium"', "10"],
    ["noise_low", '"Noise-low"', "7"],
    ["noise_tiny", '"Noise-tiny"', "3"],
    ["noise_high", '"Noise-high"', "17"]
]

path_to_script = '/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/get_pulse.py'
path_to_test_wrd = '/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/test_wpd_raw_auto.py'
path_to_python = '/usr/local/bin/python3.7'
root = "/Users/maksimrepp/PycharmProjects/webcam-pulse-detector/"

for settings in vers:
    path_to_original = "%sv.mp4" % root
    path_to_video = "%sv_%s.mp4" % (root, settings[0])

    # if "high" in settings[0]:
    #     bashCommand = 'ffmpeg -i %s -vf noise=c0s=%s:allf=t,format=yuv420p %s -y' % (path_to_original, settings[2], path_to_video)
    #     print(bashCommand)
    #
    #     process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    #     output, error = process.communicate()
    #     print(output)

    sys.argv[0] = '-v %s' % path_to_video
    bashCommand = '%s %s -f %s -g %s' % (path_to_python, path_to_test_wrd, settings[0], settings[1])
    print(bashCommand)

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
