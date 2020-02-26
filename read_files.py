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