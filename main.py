import data_extraction
import processing
import analysis
import numpy as np


def preprocess(data):
    filtered = processing.bandpass_filter(data, 100, 1, order=1)
    if len(data.shape) >= 2:
        preprocessed = processing.bwr_signals(filtered)
    else:
        preprocessed = processing.bwr(filtered)
    return preprocessed


def extract_fecg(data, qrs=None):
    # FastICA - TS
    ica1 = analysis.fast_ica(data, 4, processing.tanh)
    # processing.plot_record(ica1, time_range=(0, 0.02),
    #                        title=["<b>Independent component: {}<b>".format(str(x)) for x in range(data.shape[0])],
    #                        xlabel='<b>Time, s<b>',
    #                        ylabel='<b>Amplitude<b>')
    r_peaks = analysis.find_qrs(ica1[0, :], peak_search='original')
    r_peaks = analysis.peak_enhance(ica1[0, :], peaks=r_peaks, window=0.3)
    processing.bwr_signals(ica1)
    subtracted = analysis.ts_method(ica1, peaks=r_peaks, template_duration=0.6, fs=processing.FS, window=10)
    # processing.scatter_beautiful(subtracted[3, :], time_range=(0, 1),
    #                              title='<b>Independent component 1',
    #                              xlabel='<b>Time (s)<b>',
    #                              ylabel='<b>Amplitude (V)<b>')
    #
    # processing.plot_record(subtracted, time_range=(0, 0.02),
    #                        title=["Independent component: {}".format(str(x)) for x in range(subtracted.shape[0])],
    #                        xlabel='Time, s',
    #                        ylabel='Amplitude'
    #                        )
    ica2 = analysis.fast_ica(subtracted, 4, processing.tanh)
    if np.max(np.abs(subtracted[0, :])) < np.max(np.abs(subtracted[1, :])):
        fecg = subtracted[1, :]
    else:
        fecg = subtracted[0, :]
    # processing.plot_record(ica2, time_range=(0, 0.1),
    #                        title=["Independent component: {}".format(str(x)) for x in range(subtracted.shape[0])],
    #                        xlabel='Time, s',
    #                        ylabel='Amplitude'
    #                        )
    # processing.scatter_beautiful(fecg, time_range=(0, 1), title='qwerty')

    return fecg


def rr_analysis(fetal_ecg, kernel: tuple = (6,), fs: int = processing.FS):
    """
    Analysis fetal VHR

    :param fs: Sample frequency
    :param kernel: Kernel for median filtration
    :param fetal_ecg: Fetal ECG
    :return: rr_intervals, sample frequency
    """
    peaks = analysis.find_qrs(fetal_ecg, fs, peak_search='Original')
    enhanced_peaks = analysis.peak_enhance(fetal_ecg, peaks, window=0.08)
    rr_intervals, fs_rr = analysis.calculate_rr(enhanced_peaks, mode='bpm', time=True)
    med_rr = analysis.median_filtration(rr_intervals, kernel=kernel)
    # processing.scatter_beautiful(med_rr, fs=fs_rr,
    #                              title='Heart Rate Variability',
    #                              xlabel='Time, (s)',
    #                              ylabel='Heart Rate (bpm)')
    # processing.scatter_beautiful(rr_intervals, fs=fs_rr,
    #                              title='Heart Rate Variability',
    #                              xlabel='Time, (s)',
    #                              ylabel='Heart Rate (bpm)')

    return med_rr, fs_rr


if __name__ == '__main__':
    processing.FS = 1000
    data_full, _, qrs = processing.open_record_abd(record_name='r01', qrs=True)
    # processing.plot_record(data[:2, :], fs=processing.FS, qrs=qrs, time_range=(0, 0.01),
    #                        title=['Abdominal signal', 'Abdominal signal'],
    #                        xlabel='Time, (s)',
    #                        ylabel='Amplitude (uV)')
    data = preprocess(data_full[1:, :])
    data = preprocess(data)
    # processing.scatter_beautiful(data[3, :], fs=processing.FS, qrs=qrs, time_range=[0, 0.008],
    #                              title='<b>Abdominal signal<b>',
    #                              xlabel='<b>Time (s)<b>',
    #                              ylabel='<b>Amplitude (V)<b>')
    # processing.plot_record(data[:, :], time_range=(0, 1), qrs=qrs)
    # preprocessed = preprocess(data[1:, 0:150000])
    f_ecg = extract_fecg(data, qrs)
    pred, fs = rr_analysis(f_ecg, kernel=(6,))

    data_real = preprocess(data_full[0, :])
    real, fs2 = analysis.calculate_rr(qrs * 1000, mode='bpm', time=True)
    real = analysis.median_filtration(real, kernel=(4,))
    processing.double_Plot(pred, real, fs=(fs, fs2),
                           title="<b>Fetal Heart Rate Variability<b>",
                           xlabel='<b>Time (s)<b>',
                           ylabel='<b>Heart Rate (bpm)<b>')

    #########
    # rr_toco = processing.open_record_fhr('fhr_toco27.npy')
    # rr_intervals = analysis.median_filtration(rr_toco[0, :], (4,))
    # toco = rr_toco[1, :]
    # processing.scatter_beautiful(rr_intervals, fs=4,
    #                              title='Fetal Heart Rate',
    #                              xlabel='Time (s)',
    #                              ylabel='Heart Rate (bpm)')
    # processing.plot_record(np.vstack([rr_intervals, toco]), time_range=(0, 1),
    #                        title=["Signal channel: {}".format(str(x)) for x in range(rr_intervals.shape[0])],
    #                        xlabel='Time, s',
    #                        ylabel='Amplitude')
    # rr_intervals_ms = processing.bpm2sec(rr_intervals)
    # print(analysis.calculate_time_features(rr_intervals))
    # b, vsr_amp, acel, decel = analysis.find_signal_morphology(rr_intervals)
    # print(len(acel), len(decel), vsr_amp)
    # print(analysis.fhr_decision(rr_intervals, fs=4, acel_decel_num=True, severe=True))
#########
# print(analysis.evaluation_metrics(peaks))
# print(analysis.calculate_time_features(rr_intervals=rr_intervals))
# analysis.find_signal_morphology(rr_intervals, fs=fs)
# rr_intervals, fs = analysis.calculate_rr(qrs[0:385] * 1000, mode='bpm', time=True)
# processing.scatter_beautiful(rr_intervals, fs=fs,
#                              title='Heart Rate Variability',
#                              xlabel='Time, (s)',
#                              ylabel='Heart Rate (bpm)')

# fhr_toco = processing.open_record_fhr()
# med_rr = analysis.median_filtration(fhr_toco[1, :], kernel=(4,))
# print(analysis.calculate_time_features(processing.bpm2sec(rr_intervals), limits=(450, 500)))
# print(analysis.find_signal_morphology(rr_intervals, fs))
# processing.scatter_beautiful(med_rr, fs=4,
#                              title='Heart Rate Variability',
#                              xlabel='Time, (s)',
#                              ylabel='Heart Rate (bpm)')

# data = extract_fecg(data)
# _, data_wo_drift = processing.bwr(data[0])
# filtered = processing.bandpass_filter(data_wo_drift, 70, 0.05, 5)
# processing.scatter_beautiful(data[0], fs=processing.FS, time_range=(0, 0.02), spectrum=False,
#                              title='<b>Fetal scalp ECG <b>',
#                              xlabel='<b>Time (s)<b>',
#                              ylabel='<b>Amplitude (V)<b>')
# processing.scatter_beautiful(data_wo_drift, fs=processing.FS, time_range=(0, 0.02), spectrum=False,
#                              title='<b>Fetal scalp ECG after baseline removal<b>',
#                              xlabel='<b>Time (s)<b>',
#                              ylabel='<b>Amplitude (V)<b>')
# processing.scatter_beautiful(filtered, fs=processing.FS, time_range=(0, 0.02), spectrum=False,
#                              title='<b>Fetal scalp ECG after baseline removal<b>',
#                              xlabel='<b>Time (s)<b>',
#                              ylabel='<b>Amplitude (V)<b>')

# data = preprocess(data)
# # data = processing.open_record_DaISy()
# processing.plot_record(data[:, 0:10000], time_range=(0, 1), qrs=qrs[:380])
# extract_fecg(data[1:])
# data = processing.bandpass_filter(data, 60, 1)

# data = processing.bwr_signals(data[:, 0:10000])
# processing.scatter_beautiful(data[0], fs=processing.FS, time_range=(0, 1), spectrum=False,
#                              title='<b>Abdominal ECG<b>',
#                              xlabel='<b>Time (s)<b>',
#                              ylabel='<b>Amplitude (V)<b>')
# processing.scatter_beautiful(base, fs=processing.FS, time_range=(0, 0.01), spectrum=False,
#                              title='<b>Abdominal ECG<b>',
#                              xlabel='<b>Time (s)<b>',
#                              ylabel='<b>Amplitude (uV)<b>')
# processing.plot_record(data[:5], time_range=(0, 1))
# signal = analysis.ts_method(data[:5], template_duration=0.08)
# sources = analysis.fast_ica(data[0:5, :], 4, func=processing.tanh)
#
# processing.plot_record(signal, time_range=(0, 1))
# processing.plot_fft(sources, freq_range=(0, 0.5), mwa_window=10)
# processing.scatter_beautiful(sources[0, :], fs=processing.FS, time_range=(0, 0.1), spectrum=True,
#                              title='<b>Abdominal ECG<b>',
#                              xlabel='<b>Time (s)<b>',
#                              ylabel='<b>Amplitude (uV)<b>')
# qrs_fir, _ = analysis.filter_templates(processing.FS, 2)
# f_data = processing.matched_filter(data, qrs_fir[:, 0])
# qrs_calc = np.array(analysis.find_qrs(f_data[0, :]))

# print(qrs[:20] * processing.FS, qrs_calc[:20], np.average(np.power(qrs * processing.FS-qrs_calc, 2)))
# processing.plot_record(f_data,  time_range=(0, 5))
# processing.plot_fft(data, freq_range=(0, 0.5))
