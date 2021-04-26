import data_extraction
import processing
import analysis
import numpy as np

if __name__ == '__main__':
    processing.FS = 250
    # data, _, qrs = processing.open_record_abd(qrs=True)
    data = processing.open_record_DaISy()
    #
    # data = processing.bandpass_filter(data, 10, 5)
    processing.scatter_beautiful(data[1], fs=processing.FS, time_range=(0, 0.25), spectrum=False,
                                 title='<b>Abdominal ECG<b>',
                                 xlabel='<b>Time (s)<b>',
                                 ylabel='<b>Amplitude (uV)<b>')
    sources = analysis.fast_ica(data[0:4, :], 4, func=processing.tanh)
    processing.plot_record(sources, time_range=(0, 0.25))

    # qrs_fir, _ = analysis.filter_templates(processing.FS, 2)
    # f_data = processing.matched_filter(data, qrs_fir[:, 0])
    # qrs_calc = np.array(analysis.find_qrs(f_data[0, :]))

    # print(qrs[:20] * processing.FS, qrs_calc[:20], np.average(np.power(qrs * processing.FS-qrs_calc, 2)))
    # processing.plot_record(f_data,  time_range=(0, 5))
    # processing.plot_fft(data, freq_range=(0, 0.5))
