import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_extraction import FOLDERS
from scipy.signal import butter, iirnotch, sosfilt, lfilter, filtfilt
from scipy.signal.windows import hann
from scipy.fft import fft, fftfreq

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
    n_col = int(np.floor(data_shape[0] / n_row))
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


def scatter_beautiful(data, fs, time_range: tuple = (0, 1), spectrum: bool = False, **kwargs):
    """Plots go.Scatter with title, axis and so on.

    kwargs: 'title'; 'xlabel'; 'ylabel'."""
    data = data[int(len(data) * time_range[0]):int(len(data) * time_range[1])]
    N = len(data)
    if spectrum:
        time = fftfreq(N, 1 / FS)[int(N // 2 * time_range[0]):
                                  int(N // 2 * time_range[1])]
        window = hann(N)
        data = fft(data * window)
    else:
        time = np.arange(0, (N * 1 / fs) - 1 / fs, 1 / fs)
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
    fig.update_xaxes(title=dict(text=kwargs['xlabel'], font=dict(size=25, color="Black")))
    fig.update_yaxes(title=dict(text=kwargs['ylabel'], font=dict(size=25, color="Black")))
    fig.show()
    return None


def notch_filter(data, cutoff):
    """Filters signal with given parameters"""
    c_freq = 2 * cutoff / FS
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
    sos = butter(order, low_freq, btype='lowpass', fs=FS, output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data


def highpass_filter(data, order, high_freq):
    sos = butter(order, high_freq, btype='highpass', fs=FS, output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data


def bandpass_filter(data, high, low, order: int = 3):
    sos = butter(order, [low, high], btype='band', fs=FS, output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data


def matched_filter(data, b):
    """b - fir coefficients (template)"""
    # filtered_data = lfilter(b, 1, data)
    filtered_data = filtfilt(b, 1, data)
    return filtered_data


if __name__ == '__main__':
    data = open_record_DaISy()
    print(data.shape)
