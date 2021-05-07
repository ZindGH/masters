import data_extraction
import processing
import analysis
import numpy as np


def extract_fecg(data):
    data_bwr = processing.bwr_signals(data)
    # FastICA - TS - FastICA
    ica1 = analysis.fast_ica(data_bwr[:, 0:10000], 4, processing.tanh)
    processing.plot_record(ica1, time_range=(0, 1), qrs=qrs[:18])
    subtracted = analysis.ts_method(ica1, template_duration=0.12, fs=processing.FS)
    processing.plot_record(subtracted, time_range=(0, 1), qrs=qrs[:18])
    ica2 = analysis.fast_ica(subtracted, 4, processing.tanh)
    processing.plot_record(ica2, time_range=(0, 1))

    return None


if __name__ == '__main__':
    processing.FS = 1000
    data, _, qrs = processing.open_record_abd(qrs=True)
    # data = processing.open_record_DaISy()
    processing.plot_record(data[:, 0:10000], time_range=(0, 1), qrs=qrs[:18])
    extract_fecg(data[1:])
    # data = processing.bandpass_filter(data, 60, 1)
    # processing.scatter_beautiful(data[0], fs=processing.FS, time_range=(0, 0.01), spectrum=False,
    #                              title='<b>Abdominal ECG<b>',
    #                              xlabel='<b>Time (s)<b>',
    #                              ylabel='<b>Amplitude (uV)<b>')
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
