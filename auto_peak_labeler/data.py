import os
import pickle
import numpy as np

from .utils import get_patch_list


class NpyDataSource:

    def __init__(self, path_img, path_peaks, win_size = 10):
        self.path_img   = path_img
        self.path_peaks = path_peaks
        self.win_size   = win_size


    def load_patch_list(self):
        win_size = self.win_size

        img   = np.load(self.path_img)
        peaks = np.load(self.path_peaks)

        peaks_y = np.array([y for y, x in peaks])
        peaks_x = np.array([x for y, x in peaks])

        patch_list = get_patch_list(peaks_y, peaks_x, img, win_size = win_size)

        return patch_list
