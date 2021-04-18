import data_extraction
import processing
import analysis
import numpy as np
if __name__ == '__main__':
    # data_extraction.edf2npy_save()
    data, _, qrs = processing.open_record(qrs=True)
    # data = processing.notch_filter(data, 60)
    # data = processing.bandpass_filter(data, 100, 0.05)
    # processing.FS = 25
    processing.plot_record(data, qrs=qrs, time_range=(0, 0.01))
    qrs_fir, _ = analysis.filter_templates(processing.FS, 2)
    f_data = processing.matched_filter(data, qrs_fir[:, 0])
    # qrs_calc = np.array(analysis.find_qrs(f_data))
    # print(qrs[:20] * processing.FS, qrs_calc[:20], np.average(np.power(qrs * processing.FS-qrs_calc, 2)))
    processing.plot_record(f_data,  time_range=(0, 0.01))
    # Processing.plot_fft(data, freq_range=(0, 0.02))
