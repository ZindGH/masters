from mne.io import read_raw_edf
import numpy as np
import os

FOLDERS = {'abd': 'abd2012',
           'arr': 'arr_nr2019',
           'DaISy': 'DaiSy',
           'FHR': 'fhr'}


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


def fhr2npy_save(folder_name: str = FOLDERS['FHR']):
    """   Fetal heart rate signal dataset for training morphological analysis
     methods and evaluating them against an expert consensus
     !!Sample frequency is 4 Hz for both FHR and UA!!

    :return:
    """
    new_folder = folder_name + '_npy'
    if not os.path.isdir(new_folder):
        os.mkdir(new_folder)
    if not folder_name.endswith('/'):
        folder_name += '/'
    for i, name in enumerate(os.listdir(folder_name)):
        if name.endswith('.fhr'):
            file = open(file=folder_name + name, mode='rb')
            toco = np.fromfile(file, dtype=np.uint8, offset=3)[5::6].reshape((-1, 1)) / 2
            file = open(file=folder_name + name, mode='rb')
            fhr = np.fromfile(file, dtype=np.uint16)[2::3].reshape((-1, 1)) / 4
            fhr_toco = np.concatenate((fhr, toco), axis=1)
            np.save(new_folder + '/' + 'fhr_toco' + str(i), fhr_toco)
    return None


if __name__ == '__main__':
    fhr2npy_save(folder_name='fhr')
    # space_separ2np()
