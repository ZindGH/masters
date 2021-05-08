import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, iirnotch, sosfilt, lfilter, filtfilt
from sklearn.decomposition import FastICA
import processing
from scipy import interpolate


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


def find_qrs(ecg, fs: int = processing.FS, peak_search: str = 'custom'):
    """ GROUP DELAY FIX REQUIRED
    :find_peak: peak search algorithm - "original", "custom" available.
    """
    ecg = processing.bandpass_filter(ecg, high=15, low=5, order=1)
    # processing.scatter_beautiful(ecg, title="square OUTPUT")
    diff = np.diff(ecg)
    squared = diff * diff
    # Moving mean
    N = int(0.08 * fs)
    # processing.scatter_beautiful(squared, title="square OUTPUT")
    # mwa = processing.MWA(squared, N)
    mwa = processing.mwa_np(squared, window=N, mode='same')
    # processing.scatter_beautiful(mwa, title="mwa OUTPUT before final zerowing")
    # mwa[:int(0.08 * fs)] = 0
    if peak_search == "original":
        mwa_peaks = panPeakDetect(mwa, fs)
    else:
        mwa_peaks = peak_detect(mwa, int(0.2 * fs))
    # mwa_peaks -= ((N - 1) // 2)  # Group delay = window size / 2
    return mwa_peaks


def panPeakDetect(detection, fs):
    """ Algorithm is not working {or working, idc. anyway rework to numpy}"""

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

        if 0 < i < len(detection) - 1:
            if detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
                peak = i
                peaks.append(i)

                if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.25 * fs:

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
    # First possible peak detection
    first_possible_peak = np.argmax(detection[0:int(0.25 * fs)])
    if detection[first_possible_peak] > SPKI:
        signal_peaks[0] = first_possible_peak
    else:
        signal_peaks.pop(0)
    signal_peaks = np.array(signal_peaks)
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


def ts_method(signal, template_duration: float = 0.12, fs: int = processing.FS, peak_search: str = 'original',
              **kwargs):
    """
    Subtracts ECG signal from aECG with unknown template

    Parameters
    ----------
    signal :  numpy array with dimensions (n,m) where m is number of points; n - samples
              set of signals, where first is for mQRS detection
    template_duration : number of s in template (between qrs)
                        if points not odd ==> + 1
    fs : sampling frequency
    peak_search : peak search algorithm for "find_qrs" function
    """

    t_dur = round(template_duration * fs)
    if not t_dur % 2 == 0:
        t_dur += 1
    dims = signal.shape
    # if np.max(np.abs(signal[0, :])) < np.max(np.abs(signal[1, :])):
      # r_peaks = find_qrs(signal[1, :], peak_search=peak_search)
      # r_peaks = peak_enhance(signal[1, :], peaks=r_peaks, window=0.2)
    # else:
    r_peaks = find_qrs(signal[0, :], peak_search=peak_search)
    r_peaks = peak_enhance(signal[0, :], peaks=r_peaks, window=0.2)
    processing.scatter_beautiful(r_peaks * 1000 / fs, title='peaks')

    # print(len(r_peaks))
    # Please, rework it...
    for n in range(dims[0]):
        template = np.full((len(r_peaks), t_dur), np.nan)
        for num, r_ind in enumerate(r_peaks):
            if r_ind < t_dur // 2:
                template[num, t_dur // 2 - r_ind - 1:] = signal[n, 0:r_ind + t_dur // 2 + 1]
            elif r_ind + t_dur // 2 + 1 > dims[1]:
                template[num, 0:dims[1] - r_ind + t_dur // 2] = signal[n, r_ind - t_dur // 2:]
            else:
                template[num] = signal[n, r_ind - t_dur // 2:r_ind + t_dur // 2]
        template_mean = np.nanmean(template, axis=0)
        for r_ind in r_peaks:
            if r_ind < t_dur // 2:
                signal[n, 0:r_ind + t_dur // 2 + 1] -= template_mean[t_dur // 2 - r_ind - 1:]
                # processing.scatter_beautiful(components[n, :], title=' subtracted channel start ' + str(n))
            elif r_ind + t_dur // 2 + 1 > dims[1]:
                signal[n, r_ind - t_dur // 2:r_ind + t_dur // 2 + 1] -= template_mean[
                                                                        0:dims[1] - r_ind + t_dur // 2]
                # processing.scatter_beautiful(components[n, :], title=' subtracted channel end ' + str(n))
            else:
                signal[n, r_ind - t_dur // 2:r_ind + t_dur // 2] -= template_mean
                # processing.scatter_beautiful(components[n, :], title=' subtracted channel ' + str(n))
    return signal


def peak_enhance(signal, peaks, window: int = 0.08, fs: int = processing.FS):
    """Enhanced peaks with maximum amplitude centering
    :signal: ECG signal
    :peaks: detected peaks
    :window: centering window"""
    window = int(fs * window)
    if not window % 2 == 0:
        window += 1
    enhanced_peaks = np.zeros(len(peaks), dtype=int)
    for i, peak in enumerate(peaks):
        if peak < window // 2:
            enhanced_peaks[i] = np.argmax(signal[0:peak + window // 2 + 1])
        elif peak + window // 2 + 1 > signal.shape[0]:
            enhanced_peaks[i] = np.argmax(signal[peak - window // 2:]) + peak - window // 2
        else:
            # Because of one-side lag -> window: p - w * 0.25% : p + w * 75%
            enhanced_peaks[i] = np.argmax(signal[peak - window // 4:peak + 3 * window // 4]) + peak - window // 4

    return enhanced_peaks


def cubic_interpolation(signal, multiplier: int = 2, fs: int = processing.FS, time: bool = True):
    """


    :param signal: 1D R-peaks data
    :param multiplier: final data len is len(signal) * multiplier
    :param time: if output consists of time values
    :param fs: Sampling frequency
    :return: Interpolated signal

    """

    N = len(signal)
    x = np.arange(N * multiplier)
    interpolated = interpolate.CubicSpline(range(N), signal)
    if time:
        statement=1
    else:
        return interpolated(x)

def calculate_rr(peaks, mode: str = "sec", fs: int = processing.FS):
    """
    Calculate RR intervals from peaks

    :param peaks: peak indexes
    :param mode: output mode: "sec" (in seconds), "bpm" (beats per minute)
    :param fs: sampling frequency
    :return: Heart rate
    """
    size = len(peaks) - 1
    rr_intervals = np.ndarray(size, dtype=int)
    for i in range(size):
        rr_intervals[i] = np.abs(peaks[i] - peaks[i + 1])
    if mode == 'sec':
        rr_intervals *= 1000/fs
    elif mode == 'bmp':
        rr_intervals = rr_intervals / 60
    return rr_intervals

if __name__ == '__main__':
    templ, _ = filter_templates(processing.FS, 3)
    qrs_peaks = find_qrs()
    # Plot template
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=templ[:, 1], y=templ[:, 0]))
    fig.show()
