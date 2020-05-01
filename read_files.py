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

def read_file_mio(file):
    csv = pd.read_csv(file).to_numpy()
    times, bpms = csv[:, 0], csv[:, 1]
    return times, bpms

