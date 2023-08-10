#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import lmfit
import pickle

from auto_peak_labeler.data import NpyDataSource
from auto_peak_labeler.app  import MPIPseudoVoigt2DLabeler

mpi_batch_size = 1
path_img       = "auto_labeler.data.img.npy"
path_peaks     = "auto_labeler.data.peaks.npy"
win_size       = 10

data_source = NpyDataSource(path_img, path_peaks, win_size = 10)

with MPIPseudoVoigt2DLabeler(data_source, mpi_batch_size) as labeler:
    res_list = labeler.fit()

    if res_list is not None:
        path_out = f"auto_labeler.data.out.mpi.pickle"
        with open(path_out, 'wb') as handle:
            pickle.dump(res_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Done!!!")
