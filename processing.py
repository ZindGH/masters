import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_extraction import FOLDERS
from scipy.signal import butter, iirnotch, sosfilt, lfilter, filtfilt, sosfiltfilt
from scipy.signal.windows import hann
from scipy.fft import fft, fftfreq
import pywt

# You should change module FS parameter if signal has another sample frequency
# abd = 1000 Hz
# DaISy = 250 Hz
FS = 1000


def open_record_abd(record_name: str = 'r01', qrs: bool = True):
    """ Loads data, channel info and qrs timestamps """
    folder = FOLDERS['abd'] + '_npy/'
    data = np.load(folder + record_name + '_data.npy')
    channels = np.load(folder + record_name + '_ch.npy')
    if qrs:
        qrs = np.load(folder + record_name + '_QRS.npy')
        return data, channels, qrs
    return data, channels


def open_record_DaISy(record_name: str = '/daisy.npy'):
    folder = FOLDERS['DaISy']
    data = np.load(folder + record_name).T
    return data


def plot_record(data, qrs=None, time_range: tuple = (0, 1), fft_plot: bool = False):
    """Plots all channels in different axes. With QRS points if included"""

    # Cut data for plotting
    if not fft_plot:
        data = data[:, int(data.shape[1] * time_range[0]):int(data.shape[1] * time_range[1])]
    if qrs is not None:
        qrs = qrs[int(qrs.shape[0] * time_range[0]):int(qrs.shape[0] * time_range[1])]
    data_shape = data.shape
    n_row = int(data_shape[0] // 2)
    n_col = int(np.ceil(data_shape[0] / n_row))
    if fft_plot:
        time = fftfreq(data_shape[1], 1 / FS)[int(data_shape[1] // 2 * time_range[0]):
                                              int(data_shape[1] // 2 * time_range[1])]

    else:
        time = np.arange(0, (data_shape[1] * 1 / FS) - 1 / FS, 1 / FS)
    fig = make_subplots(rows=n_row, cols=n_col)
    # Fill subplots
    k_plot = 0
    for row in range(n_row):
        for col in range(n_col):
            fig.add_trace(go.Scatter(x=time, y=data[k_plot, :]), row=row + 1, col=col + 1)
            if qrs is not None:
                fig.add_trace(go.Scatter(x=qrs, y=np.full((qrs.shape[0],), max(data[k_plot, :])),
                                         mode='markers',
                                         marker=dict(size=2)),
                              col=col + 1,
                              row=row + 1)
            k_plot += 1
            if k_plot > data_shape[0] - 1:
                break

        # for qrs_line in qrs:
        #     fig.add_vline(x=qrs_line, line_width=0.3)
        #     print(np.round(qrs_line))

    fig.show()


def scatter_beautiful(data, fs: int = FS, time_range: tuple = (0, 1), spectrum: bool = False, mwa_window=None,
                      **kwargs):
    """Plots go.Scatter with title, axis and so on.

    kwargs: 'title'; 'xlabel'; 'ylabel'.
    mwa_window = None, for window = 1 sample; 0:i:1 for fraction of sample frequency.
    """
    N = len(data)
    if spectrum:
        time = fftfreq(N, 1 / fs)[int(N // 2 * time_range[0]):
                                  int(N // 2 * time_range[1])]
        window = hann(N)
        data = fft(data * window)
        data = 2 / N * np.abs(data[0:N // 2])
        if mwa_window:
            data = MWA(data, int(mwa_window * 0.12))
    else:
        time = np.arange(0, (N * 1 / fs), 1 / fs)
        data = data[int(len(data) * time_range[0]):int(len(data) * time_range[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, x=time))
    fig.update_layout(
        title={
            'text': kwargs['title'],
            'y': 0.92,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font=dict(
            family="Times New Roman",
            size=22,
            color="Black"
        )

    )
    if 'xlabel' in kwargs:
        fig.update_xaxes(title=dict(text=kwargs['xlabel'], font=dict(size=25, color="Black")))
    if 'ylabel' in kwargs:
        fig.update_yaxes(title=dict(text=kwargs['ylabel'], font=dict(size=25, color="Black")))
    fig.show()
    return None


def notch_filter(data, cutoff):
    """Filters signal with given parameters"""
    c_freq = 2 * cutoff / FS
    b, a = iirnotch(c_freq, 30)  # Creating digital filter coefficients
    filtered_data = lfilter(b, a, data)
    return filtered_data


def plot_fft(data, freq_range: tuple = (0, 1), mwa_window=None):
    """Plot fft of numpy signal.

    Parameters
    ----------
    data :  numpy array with dimensions (n,m) where m is number of points
    freq_range : sets frequency range, (min, max) where max is m // 2
    mwa_window : window in points."""

    shape = data.shape
    window = hann(shape[1])
    yf = fft(data * window)
    # print(yf.shape)
    yf = 2 / shape[1] * np.abs(yf[0:shape[1] // 2])
    if mwa_window:
        for row in range(yf.shape[0]):
            yf[row, :] = MWA(yf[row, :], int(mwa_window))

    plot_record(yf, time_range=freq_range, fft_plot=True)


def lowpass_filter(data, order, low_freq):
    sos = butter(order, low_freq, btype='lowpass', fs=FS, output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data


def highpass_filter(data, order, high_freq):
    sos = butter(order, high_freq, btype='highpass', fs=FS, output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data


def bandpass_filter(data, high, low, order: int = 3):
    sos = butter(order, [low, high], btype='band', fs=FS, output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data


def matched_filter(data, b):
    """b - fir coefficients (template)"""
    # filtered_data = lfilter(b, 1, data)
    filtered_data = filtfilt(b, 1, data)
    return filtered_data


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


def whiten(data):
    data = data.T - np.mean(data.T)
    sigma = np.cov(data, rowvar=True)  # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(sigma)
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCA = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))  # [M x M]
    data_whitened = np.dot(ZCA, data)
    return data_whitened.T


def tanh(y):
    return np.log(np.cosh(y)) - 0.375, np.tanh(y)


def bwr(signal):
    """
    Calculate the baseline of signal.

    Args:
        signal (numpy 1d array): signal whose baseline should be calculated


    Returns:
        baseline (numpy 1d array with same size as signal): baseline of the signal
    """
    ssds = np.zeros((3))

    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return baseline[: len(signal)], signal - baseline[: len(signal)]


def bwr_signals(signal):
    """ Returns BWR of all signals with format [n_channel, samples]"""
    for row in range(signal.shape[0]):
        _, signal[row, :] = bwr(signal[row, :])
    return signal


def mwa_np(data, window: int = 40, lag: bool = True, **kwargs):
    if 'mode' in kwargs:
        mwa = np.convolve(data, np.ones(window), mode=kwargs['mode']) / window
    else:
        mwa = np.convolve(data, np.ones(window)) / window
    return mwa


if __name__ == '__main__':
    data = open_record_DaISy()
    print(data.shape)
