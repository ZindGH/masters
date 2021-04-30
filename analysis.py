import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, iirnotch, sosfilt, lfilter, filtfilt
from sklearn.decomposition import FastICA
import processing


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
    f1 = 5 / fs
    f2 = 15 / fs
    b, a = butter(1, [f1 * 2, f2 * 2], btype='bandpass')
    # Butter +
    filtered_ecg = filtfilt(b, a, ecg)
    diff = np.diff(filtered_ecg)
    squared = diff * diff
    # Moving mean
    N = int(0.12 * fs)
    mwa = processing.MWA(squared, N)
    mwa[:int(0.2 * fs)] = 0

    # mwa_peaks = panPeakDetect(mwa, fs)
    mwa_peaks = peak_detect(mwa, int(0.4 * fs))
    return mwa_peaks


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


def peak_detect(data, spacing=1, limit=None):
    """
    Algorithm was borrowed from here:
    https://github.com/jankoslavic/py-tools/blob/master/findpeaks/findpeaks.py

    Finds peaks in `data` which are of `spacing` width and >=`limit`.
    :param data: values
    :param spacing: minimum spacing to the next peak (should be 1 or more)
    :param limit: peaks should have value greater or equal
    :return: indexes
    """
    ln = data.size
    x = np.zeros(ln + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + ln] = data
    peak_candidate = np.zeros(ln)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + ln]  # before
        start = spacing
        h_c = x[start: start + ln]  # central
        start = spacing + s + 1
        h_a = x[start: start + ln]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind


def fast_ica(x, n: int = 3, func='cube'):
    """ Implement FastICA with 'n' sources
    ____________
    func: 'cube', 'logcosh', 'exp' OR function with (G(y), g(x)), where g=G'
    """
    fastica = FastICA(n, algorithm='deflation', fun=func)
    return fastica.fit_transform(x.T).T


def ts_method(signal, template_duration: float = 0.45, fs: int = processing.FS):
    """
    Subtracts ECG signal from aECG with unknown template

    Parameters
    ----------
    signal :  numpy array with dimensions (n,m) where m is number of points; n - samples
              set of signals, where first is for mQRS detection
    template_duration : number of s in template (between qrs)
                        if points not odd ==> + 1
    fs : sampling frequency
    """

    t_dur = round(template_duration * fs)
    if not t_dur % 2 == 0:
        t_dur += 1
    dims = signal.shape
    components = fast_ica(signal, 4, processing.tanh)
    # components = signal
    r_peaks = find_qrs(components[0, :])

    # Please, rework it...
    for n in range(dims[0]):
        template = np.full((len(r_peaks), t_dur), np.nan)
        for num, r_ind in enumerate(r_peaks):
            if r_ind < t_dur // 2:
                template[num, t_dur // 2 - r_ind - 1:] = components[n, 0:r_ind + t_dur // 2 + 1]
            elif r_ind + t_dur // 2 + 1 > dims[1]:
                template[num, 0:dims[1] - r_ind + t_dur // 2] = components[n, r_ind - t_dur // 2:]
            else:
                template[num] = components[n, r_ind - t_dur // 2:r_ind + t_dur // 2]
        template_mean = np.nanmean(template, axis=0)
        for r_ind in r_peaks:
            if r_ind < t_dur // 2:
                components[n, 0:r_ind+t_dur//2+1] -= template_mean[t_dur // 2 - r_ind:]
            elif r_ind + t_dur // 2 + 1 > dims[1]:
                components[n, r_ind-t_dur//2:r_ind+t_dur//2+1] -= template_mean[0:dims[1] - r_ind + t_dur // 2]
            else:
                components[n, r_ind-t_dur//2:r_ind+t_dur//2] -= template_mean

    return components


if __name__ == '__main__':
    # templ, _ = filter_templates(processing.FS, 3)
    # qrs_peaks = find_qrs()
    # # Plot template
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=templ[:, 1], y=templ[:, 0]))
    fig.show()
