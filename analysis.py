import numpy as np
from processing import FS
import plotly.graph_objects as go
from ecgdetectors import Detectors
from scipy.signal import butter, iirnotch, sosfilt, lfilter, filtfilt


def filter_templates(fs: int, qrs_template: int = 1):
    """Creates fetal QRS and PQRST templates from [22]
    qrs: (1) h' = -0.007^2 * t * e^(-t^2/0.007^2)
         (2) h = e^(-t^2/0.007^2)
         (3) h3 = −0.2 * e^(-(t + 0.007)^2 / 0.005^2)
                  +e^(-t^2 / 0.005^2)
                  -0.2 * e^(-(t - 0.007)^2 / 0.005^2)"""

    N = int(0.04 * fs + 1)
    qrs = np.zeros((N, 2))
    qrs[:, 1] = np.arange(-0.02, 0.02 + 1 / fs, 1 / fs)  # Time

    ### Templates
    if qrs_template == 1:
        qrs[:, 0] = -2 * np.exp(-np.power(qrs[:, 1], 2) / np.power(0.007, 2)) * qrs[:, 1] / (np.power(0.007, 2))
    elif qrs_template == 2:
        qrs[:, 0] = np.exp(-np.power(qrs[:, 1], 2) / np.power(0.007, 2))
    else:
        qrs[:, 0] = - 0.2 * np.exp(-np.power(qrs[:, 1] + 0.007, 2) / np.power(0.005, 2)) \
                    + np.exp(-np.power(qrs[:, 1], 2) / np.power(0.005, 2)) \
                    - 0.2 * np.exp(-np.power(qrs[:, 1] - 0.007, 2) / np.power(0.005, 2))
    ###

    qrs[:, 0] = (qrs[:, 0] - qrs[:, 0].mean()) / (qrs[:, 0].std())  # Normalization
    qrs[:, 1] = qrs[::-1, 1]  # Reversing => fir_coeffs(b)
    #########
    pqrst = None
    return qrs, pqrst


def find_qrs(ecg, fs: int = 1000):
    # GROUP DELAY FIX REQUIRED #
    # detector = Detectors(FS)
    # qrs_peaks = detector.pan_tompkins_detector(data[0, :])
    f1 = 5 / fs
    f2 = 15 / fs
    b, a = butter(1, [f1 * 2, f2 * 2], btype='bandpass')
    # Butter +
    filtered_ecg = filtfilt(b, a, ecg)
    diff = np.diff(filtered_ecg)
    squared = diff * diff
    # Moving mean
    N = int(0.12 * fs)
    mwa = MWA(squared, N)
    mwa[:int(0.2 * fs)] = 0

    mwa_peaks = panPeakDetect(mwa, fs)

    return mwa_peaks


def MWA(input_array, window_size):
    """Moving mean from ecgdetectors.py
    window_size: fs*0.12(recommended)"""
    mwa = np.zeros(len(input_array))
    for i in range(len(input_array)):
        if i < window_size:
            section = input_array[0:i]
        else:
            section = input_array[i - window_size:i]

        if i != 0:
            mwa[i] = np.mean(section)
        else:
            mwa[i] = input_array[i]

    return mwa


def panPeakDetect(detection, fs):
    min_distance = int(0.25 * fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):

        if i > 0 and i < len(detection) - 1:
            if detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
                peak = i
                peaks.append(i)
                # 0.3 * fs - N difference between peaks
                if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.2 * fs:

                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                    if RR_missed != 0:
                        if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                            missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[
                                    -1] - missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2) > 0:
                                missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

                threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                threshold_I2 = 0.5 * threshold_I1

                if len(signal_peaks) > 8:
                    RR = np.diff(signal_peaks[-9:])
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66 * RR_ave)

                index = index + 1

    signal_peaks.pop(0)

    return signal_peaks


if __name__ == '__main__':
    templ, _ = filter_templates(FS, 3)
    qrs_peaks = find_qrs()
    # Plot template
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=templ[:, 1], y=templ[:, 0]))
    fig.show()
