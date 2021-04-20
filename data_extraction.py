from mne.io import read_raw_edf
import numpy as np
import os

FOLDERS = {'abd': 'abd2012',
           'arr': 'arr_nr2019',
           'DaISy': 'DaiSy'}


def space_separ2np(path: str = 'DaISy/FOETAL_ECG.dat'):
    data = np.loadtxt(path)
    np.save(FOLDERS['DaISy'] + '/daisy.npy', data[:, 1:])
    return None


def edf2npy_save(folder_name: str = FOLDERS['abd'],
                 save_qrs: bool = True):
    """Saves .edf files as numpy arrays:
    _data.npy - raw data
    _QRS.npy - QRS time moments (annotations.onset)
    _ch.npy - channel names"""

    new_folder = folder_name + '_npy'
    os.mkdir(new_folder)
    if not folder_name.endswith('/'):
        folder_name += '/'
    for name in os.listdir(folder_name):
        if name.endswith('.edf'):
            data = read_raw_edf(folder_name + name)
            raw_data = data.get_data()
            channels = data.ch_names
            new_name = os.path.splitext(name)[0]
            np.save(new_folder + '/' + new_name + '_data.npy', raw_data)
            np.save(new_folder + '/' + new_name + '_ch.npy', np.array(channels))
            if save_qrs:
                np.save(new_folder + '/' + new_name + '_QRS.npy', data.annotations.onset)
    return None


if __name__ == '__main__':
    space_separ2np()
