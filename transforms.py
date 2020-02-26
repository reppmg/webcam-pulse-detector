import numpy as np
from scipy.signal import savgol_filter

from constants import percentile_neighbours, savgol_window, savgol_power


def transform(times, pulse):
    if len(pulse) < 302:
        return times, pulse, "too short"
    percentile = np.percentile(pulse, 90)
    percentile_low = np.percentile(pulse, 10)
    mean_window_size = percentile_neighbours
    for i, p in enumerate(pulse):
        if p > percentile:
            start = max(0, i - mean_window_size)
            end = min(len(pulse), i + mean_window_size)
            pulse[i] = np.mean(pulse[start:end])
        if p < percentile_low:
            start = max(0, i - mean_window_size)
            end = min(len(pulse), i + mean_window_size)
            pulse[i] = np.mean(pulse[start:end])
    pulse = savgol_filter(pulse, savgol_window, savgol_power)
    return times, pulse[0:len(times)], "percMeanNeigh30 + savgol cutlow 2"