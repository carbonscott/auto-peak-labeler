#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import lmfit
import pickle
import random

from auto_peak_labeler.data import MPICxiDataSource
from auto_peak_labeler.app  import MPIPseudoVoigt2DLabeler

mpi_batch_size = 1
## path_cxi       = "mfx13016/cwang31/psocake/r0038/mfx13016_0038.cxi"
path_cxi       = "mfx13016/cwang31/psocake/r0031/mfx13016_0031_22_pfmask.cxi"
win_size       = 10

with MPIPseudoVoigt2DLabeler(mpi_batch_size) as labeler:
    with MPICxiDataSource(path_cxi, win_size = win_size) as data_source:
        # Here events means hit events
        num_events = len(data_source)
        event_idx  = random.choice(range(num_events))

        data, peaks_y, peaks_x, patch_list = data_source.load_patch_list(event_idx)
        res_list = labeler.fit(patch_list)

        if data is not None:
            path_out = f"auto_labeler.{event_idx:06d}.data.npy"
            np.save(path_out, data)
            print(f"{path_out} is saved!!!")

        if peaks_y is not None and peaks_x is not None:
            path_out = f"auto_labeler.{event_idx:06d}.peaks.npy"
            peaks = np.array([(y, x) for y, x in zip(peaks_y, peaks_x)])
            np.save(path_out, peaks)
            print(f"{path_out} is saved!!!")

        if res_list is not None:
            path_out = f"auto_labeler.{event_idx:06d}.pickle"
            with open(path_out, 'wb') as handle:
                pickle.dump(res_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"{path_out} is saved!!!")
