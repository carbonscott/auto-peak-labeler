#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''

This program will read an cxi file generated by psocake's peak finding process,
work on fitting all found peaks with the pseudo voigt 2D model (profile) and
then save the labeled mask to a new 'segmask' dataset in the same cxi file.

'''

import os
import random
import numpy as np
import h5py
import time
import yaml
import argparse

from auto_peak_labeler.app   import PseudoVoigt2DLabeler
from auto_peak_labeler.utils import apply_mask, get_patch_list

from auto_peak_labeler.modeling.pseudo_voigt2d import PseudoVoigt2D


# Set up MPI
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()
mpi_data_tag = {
##    "num_batch" : 11,
    "input"     : 12,
    "output"    : 13,
    "signal"    : 21,
    "debug"     : 31,
    "sync"      : 41,
}
START_SIGNAL    = 0
TERMINAL_SIGNAL = -1


# Initialize labeler...
labeler = PseudoVoigt2DLabeler()

if mpi_rank == 0:
    # [[[ ARG PARSE ]]]
    parser = argparse.ArgumentParser(description='Process a yaml file.')
    parser.add_argument('yaml', help='The input yaml file.')
    args = parser.parse_args()

    # [[[ CONFIGURE BY YAML ]]]
    fl_yaml = args.yaml
    basename_yaml = fl_yaml[:fl_yaml.rfind('.yaml')]

    # Load the YAML file
    with open(fl_yaml, 'r') as fh:
        config = yaml.safe_load(fh)
    path_cxi_list  = config['cxi']
    win_size       = config['win_size']
    mpi_batch_size = config['mpi_batch_size']

    # Define the keys used below...
    CXI_KEY = { 
        "num_peaks"  : "/entry_1/result_1/nPeaks",
        "data"       : "/entry_1/data_1/data",
        "mask"       : "/entry_1/data_1/mask",
        "peak_y"     : "/entry_1/result_1/peakYPosRaw",
        "peak_x"     : "/entry_1/result_1/peakXPosRaw",
        "segmask"    : "/entry_1/data_1/segmask",
    }

    # Set up the threshold for peak labeling...
    sigma_level        = 2
    frac_redchi        = 0.5
    max_goodness_score = 0.5

    # Go through each cxi...
    for path_cxi in path_cxi_list:
        if not os.path.exists(path_cxi): continue
        with h5py.File(path_cxi, 'a') as fh:
            # Obtain the number of peaks per event...
            k = CXI_KEY['num_peaks']
            num_peaks_by_event = fh.get(k)

            # Allow to create a segmask dataset for each cxi file...
            creates_segmask_dataset = True

            # Go through all hit events...
            for enum_event_idx, num_peaks in enumerate(num_peaks_by_event):
                # Obtain the diffraction image...
                k = CXI_KEY['data']
                img = fh.get(k)[enum_event_idx]

                # Obtain the bad pixel mask...
                k = CXI_KEY['mask']
                mask = fh.get(k)[enum_event_idx]

                # Obtain the Bragg peak positions in this event...
                k = CXI_KEY['peak_y']
                peaks_y = fh.get(k)[enum_event_idx][:num_peaks] # A fixed length array with 0 indicating no peaks, e.g.[2,3,1,..., 0,0,0]

                k = CXI_KEY['peak_x']
                peaks_x = fh.get(k)[enum_event_idx][:num_peaks]

                # Apply mask...
                img = apply_mask(img, 1 - mask, mask_value = 0)

                # Derive image patches...
                patch_list = get_patch_list(peaks_y, peaks_x, img, win_size = win_size)

                # ___/ MPI: MANAGER BROADCAST DATA \___
                # Inform all workers the number of batches to work on...
                batch_patch_list = np.array_split(patch_list, mpi_batch_size)

                # Perform model fitting for each batch...
                res_list = []
                for batch_idx, patch_list_per_batch in enumerate(batch_patch_list):
                    # Split the workfload...
                    patch_list_per_batch_per_chunk = np.array_split(patch_list_per_batch, mpi_size)

                    for i in range(1, mpi_size, 1):
                        # Ask workers to start data process...
                        mpi_comm.send(START_SIGNAL, dest = i, tag = mpi_data_tag["signal"])

                        # Send workers data for processing...
                        data_to_send = patch_list_per_batch_per_chunk[i]
                        mpi_comm.send(data_to_send, dest = i, tag = mpi_data_tag["input"])

                        # Send debug info to workers...
                        batch_size = len(patch_list_per_batch)
                        data_to_send = (enum_event_idx, batch_idx, batch_size)
                        mpi_comm.send(data_to_send, dest = i, tag = mpi_data_tag["debug"])

                    patch_list_current_rank = patch_list_per_batch_per_chunk[0]
                    print(f"E {enum_event_idx:06d}, B {batch_idx:02d}, |C| {len(patch_list_current_rank):04d}({batch_size:04d}), R {mpi_rank:03d}.", flush = True)
                    res_list_current_rank = labeler.fit_all(patch_list_current_rank) 
                    res_list.extend(res_list_current_rank)

                    for i in range(1, mpi_size, 1):
                        res_list_current_rank = mpi_comm.recv(source = i, tag = mpi_data_tag["output"])
                        res_list.extend(res_list_current_rank)

                # Manager works on "model, threshold and update"...
                # ...model
                H, W    = img.shape[-2:]
                segmask = np.zeros((H, W), dtype = int)
                ## segmask[250:250+100, 250:250+100] = 1
                for y, x, res in zip(peaks_y, peaks_x, res_list):
                    # Is it a good fit???
                    rmsd = np.sqrt((res.residual**2).mean())
                    redchi = res.redchi
                    goodness_score = (1 - frac_redchi) * rmsd + frac_redchi * redchi
                    if goodness_score > max_goodness_score: continue

                    # Find the patch to label...
                    y             = round(y)
                    x             = round(x)
                    x_min         = max(x - win_size, 0)
                    x_max         = min(x + win_size + 1, H)
                    y_min         = max(y - win_size, 0)
                    y_max         = min(y + win_size + 1, W)
                    segmask_patch = segmask[y_min:y_max, x_min:x_max]

                    # Generate a model without background...
                    pseudo_voigt2d = PseudoVoigt2D(res.params, includes_bg = False)
                    H_peak, W_peak = segmask_patch.shape[-2:]
                    Y_peak         = np.arange(0, H_peak)
                    X_peak         = np.arange(0, W_peak)
                    Y, X           = np.meshgrid(Y_peak, X_peak, indexing = 'ij')
                    model_patch    = pseudo_voigt2d(Y, X)

                    filter_rule = model_patch > (model_patch.mean() + sigma_level * model_patch.std())
                    segmask_patch[filter_rule] = 1

                # Create segmask dataset if necessary...
                k = CXI_KEY['segmask']
                if creates_segmask_dataset:
                    # Create a placeholder for saving segmask...
                    if k in fh: 
                        print(f"Deleting existing {k}")
                        del fh[k]

                    num_event = len(num_peaks_by_event)
                    fh.create_dataset(k,
                                      (num_event, H, W),
                                      chunks           = (1, H, W),
                                      dtype            = 'int',
                                      compression_opts = 6,
                                      compression      = 'gzip',)

                    creates_segmask_dataset = False

                # Save the segmask for this event...
                fh[k][enum_event_idx] = segmask

    # Send termination signal...
    for i in range(1, mpi_size, 1):
        mpi_comm.send(TERMINAL_SIGNAL, dest = i, tag = mpi_data_tag["signal"])

else:
    while True:
        received_signal = mpi_comm.recv(source = 0, tag = mpi_data_tag["signal"])

        if received_signal == TERMINAL_SIGNAL: break

        patch_list_current_rank = mpi_comm.recv(source = 0, tag = mpi_data_tag["input"])

        enum_event_idx, batch_idx, batch_size = mpi_comm.recv(source = 0, tag = mpi_data_tag["debug"])

        print(f"E {enum_event_idx:06d}, B {batch_idx:02d}, |C| {len(patch_list_current_rank):04d}({batch_size:04d}), R {mpi_rank:03d}.", flush = True)
        res_list_current_rank = labeler.fit_all(patch_list_current_rank) 

        mpi_comm.send(res_list_current_rank, dest = 0, tag = mpi_data_tag["output"])

MPI.Finalize()