import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_extraction import FOLDERS
from scipy.signal import butter, iirnotch, sosfilt, lfilter
from scipy.signal.windows import hann
from scipy.fft import fft, fftfreq

folder = FOLDERS['abd'] + '_npy/'
# You should change module fs parameter if signal has another sample frequency
fs = 1000


def open_record(record_name: str = 'r01', qrs: bool = True):
    """ Loads data, channel info and qrs timestamps """
    data = np.load(folder + record_name + '_data.npy')
    channels = np.load(folder + record_name + '_ch.npy')
    if qrs:
        qrs = np.load(folder + record_name + '_QRS.npy')
        return data, channels, qrs
    return data, channels


def plot_record(data, qrs=None, time_range: tuple = (0, 1), fft_plot: bool = False):
    """Plots all channels in different axes. With QRS points if included"""

    # Cut data for plotting
    if not fft_plot:
        data = data[:, int(np.float(data.shape[1] * time_range[0])):int(np.float(data.shape[1] * time_range[1]))]
    if qrs is not None:
        qrs = qrs[int(np.float(qrs.shape[0] * time_range[0])):int(np.float(qrs.shape[0] * time_range[1]))]
    data_shape = data.shape
    n_row = int(data_shape[0] // 2)
    n_col = int(np.ceil(data_shape[0] / 2))
    if fft_plot:
        time = fftfreq(data_shape[1], 1 / fs)[int(np.float(data_shape[1] // 2 * time_range[0])):
                                              int(np.float(data_shape[1] // 2 * time_range[1]))]
    else:
        time = np.arange(0, (data_shape[1] * 1 / fs) - 1 / fs, 1 / fs)
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


def notch_filter(data, cutoff):
    """Filters signal with given parameters"""
    c_freq = 2 * cutoff / fs
    b, a = iirnotch(c_freq, 30)  # Creating digital filter coefficients
    filtered_data = lfilter(b, a, data)
    return filtered_data


def plot_fft(data, freq_range: tuple = (0, 1)):
    """Plot fft of numpy signal.

    Parameters
    ----------
    data :  numpy array with dimensions (n,m) where m is number of points
    freq_range : sets frequency range, (min, max) where max is m // 2. """

    shape = data.shape
    window = hann(shape[1])
    yf = fft(data * window)
    # print(yf.shape)
    plot_record(2 / shape[1] * np.abs(yf[0:shape[1] // 2]), time_range=freq_range, fft_plot=True)


def lowpass_filter(data, order, low_freq):
    sos = butter(order, low_freq, btype='lowpass', fs=fs, output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data


def highpass_filter(data, order, high_freq):
    sos = butter(order, high_freq, btype='highpass', fs=fs, output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data


def bandpass_filter(data, high, low, order: int = 3):

    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data
