import data_extraction
import processing
import analysis
import numpy as np

if __name__ == '__main__':
    # processing.FS = 250
    data, _, qrs = processing.open_record_abd(qrs=True)
    # data = processing.open_record_DaISy()
    #
    # data = processing.bandpass_filter(data, 60, 1)
    # processing.scatter_beautiful(data[0], fs=processing.FS, time_range=(0, 0.01), spectrum=False,
    #                              title='<b>Abdominal ECG<b>',
    #                              xlabel='<b>Time (s)<b>',
    #                              ylabel='<b>Amplitude (uV)<b>')
    data = processing.bwr_signals(data[:, 5000:10000])
    # processing.scatter_beautiful(data[0], fs=processing.FS, time_range=(0, 0.01), spectrum=False,
    #                              title='<b>Abdominal ECG<b>',
    #                              xlabel='<b>Time (s)<b>',
    #                              ylabel='<b>Amplitude (uV)<b>')
    # processing.scatter_beautiful(base, fs=processing.FS, time_range=(0, 0.01), spectrum=False,
    #                              title='<b>Abdominal ECG<b>',
    #                              xlabel='<b>Time (s)<b>',
    #                              ylabel='<b>Amplitude (uV)<b>')
    processing.plot_record(data, time_range=(0, 1))
    signal = analysis.ts_method(data[1:], template_duration=0.08)
    # sources = analysis.fast_ica(data[0:5, :], 4, func=processing.tanh)
    #
    processing.plot_record(signal, time_range=(0, 1))
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

