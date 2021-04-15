import data_extraction
import processing
import analysis
if __name__ == '__main__':
    # data_extraction.edf2npy_save()
    data, _, qrs = processing.open_record(qrs=True)
    # data = Processing.notch_filter(data, 60)
    # data = Processing.bandpass_filter(data, 100, 0.05)
    # Processing.FS = 25
    # Processing.plot_record(data, qrs=qrs, time_range=(0, 0.02))
    qrs_fir, _ = analysis.filter_templates(processing.FS)
    f_data = processing.matched_filter(data, qrs_fir[:, 1])
    print(f_data)
    processing.plot_record(f_data, qrs=qrs, time_range=(0, 0.02))
    # Processing.plot_fft(data, freq_range=(0, 0.02))
