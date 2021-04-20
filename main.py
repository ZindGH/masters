import data_extraction
import processing
import analysis
import numpy as np

if __name__ == '__main__':
    processing.FS = 250
    # data, _, qrs = processing.open_record_abd(qrs=True)
    data = processing.open_record_DaISy()
    processing.plot_record(data, time_range=(0, 0.5))
    qrs_fir, _ = analysis.filter_templates(processing.FS, 2)
    f_data = processing.matched_filter(data, qrs_fir[:, 0])
    qrs_calc = np.array(analysis.find_qrs(f_data[0, :]))

    # print(qrs[:20] * processing.FS, qrs_calc[:20], np.average(np.power(qrs * processing.FS-qrs_calc, 2)))
    processing.plot_record(f_data,  time_range=(0, 5))
    processing.plot_fft(data, freq_range=(0, 0.5))
