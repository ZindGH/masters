import data_extraction
import Processing

if __name__ == '__main__':
    # data_extraction.edf2npy_save()
    data, _, qrs = Processing.open_record(qrs=True)
    # data = Processing.notch_filter(data, 60)
    # data = Processing.bandpass_filter(data, 100, 0.05)
    Processing.fs = 25
    Processing.plot_record(data, qrs=qrs, time_range=(0, 0.02))

    # Processing.plot_fft(data, freq_range=(0, 0.02))
