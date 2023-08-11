import os
import pickle
import numpy as np
import h5py

from mpi4py import MPI

## from peaknet.plugins import apply_mask
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




class MPINpyDataSource:
    """
    By default, all worker nodes pass None whenever message passing is required.
    """

    def __init__(self, path_img, path_peaks, win_size = 10):
        self.mpi_comm     = MPI.COMM_WORLD
        self.mpi_rank     = self.mpi_comm.Get_rank()
        self.mpi_size     = self.mpi_comm.Get_size()
        self.mpi_data_tag = 11

        self.path_img   = path_img
        self.path_peaks = path_peaks
        self.win_size   = win_size


    def load_patch_list(self):
        patch_list = None

        if self.mpi_rank == 0:
            win_size = self.win_size

            img   = np.load(self.path_img)
            peaks = np.load(self.path_peaks)

            peaks_y = np.array([y for y, x in peaks])
            peaks_x = np.array([x for y, x in peaks])

            patch_list = get_patch_list(peaks_y, peaks_x, img, win_size = win_size)

        return patch_list




class MPICxiDataSource:
    """
    By default, all worker nodes pass None whenever message passing is required.
    """

    def __init__(self, path_cxi, win_size = 10, finalizes_MPI = False):
        self.mpi_comm     = MPI.COMM_WORLD
        self.mpi_rank     = self.mpi_comm.Get_rank()
        self.mpi_size     = self.mpi_comm.Get_size()
        self.mpi_data_tag = 11

        self.path_cxi       = path_cxi
        self.win_size       = win_size
        self.finalizes_MPI  = finalizes_MPI

        self.is_finalized = False

        self.cxi_handle = None
        self.cxi_key = {
            "num_peaks"  : "/entry_1/result_1/nPeaks",
            "data"       : "/entry_1/data_1/data",
            "mask"       : "/entry_1/data_1/mask",
            "peak_y"     : "/entry_1/result_1/peakYPosRaw",
            "peak_x"     : "/entry_1/result_1/peakXPosRaw",
        }


    def finalize(self):
        self.mpi_comm.Barrier()
        if self.finalizes_MPI:
            if not self.is_finalized:
                MPI.Finalize()
                self.is_finalized = True

                if self.mpi_rank != 0: sys.exit(0)


    def close_cxi(self):
        if self.cxi_handle is not None:
            self.cxi_handle.close()


    def __enter__(self):
        if self.mpi_rank == 0:
            self.cxi_handle = h5py.File(self.path_cxi, "r")

        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_cxi()
        self.finalize()


    def get_dataset(self, key):
        dataset = None
        if self.mpi_rank == 0:
            dataset = self.cxi_handle.get(key)[()]    # !!!Watch out: it will return data not just the iterator.

            for mpi_idx in range(1, self.mpi_size, 1):
                data_to_send = dataset
                self.mpi_comm.send(data_to_send, dest = mpi_idx, tag = self.mpi_data_tag)
        else:
            dataset = self.mpi_comm.recv(source = 0, tag = self.mpi_data_tag)

        self.mpi_comm.Barrier()

        return dataset



    def __len__(self):
        if self.mpi_rank == 0:
            cxi_key = self.cxi_key["num_peaks"]
            num_peaks = len(self.cxi_handle.get(cxi_key))

            for mpi_idx in range(1, self.mpi_size, 1):
                data_to_send = num_peaks
                self.mpi_comm.send(data_to_send, dest = mpi_idx, tag = self.mpi_data_tag)
        else:
            num_peaks = self.mpi_comm.recv(source = 0, tag = self.mpi_data_tag)

        self.mpi_comm.Barrier()

        return num_peaks


    def load_patch_list(self, idx):
        data       = None
        peaks_y    = None
        peaks_x    = None
        patch_list = None
        if self.mpi_rank == 0:
            cxi_key = self.cxi_key["num_peaks"]
            num_peaks = self.cxi_handle.get(cxi_key)[idx]

            cxi_key_data = self.cxi_key["data"]
            data = self.cxi_handle.get(cxi_key_data)[idx]

            ## cxi_key_mask = self.cxi_key["mask"]
            ## mask = self.cxi_handle.get(cxi_key_mask)[()]

            cxi_key_peak_y = self.cxi_key["peak_y"]
            peaks_y = self.cxi_handle.get(cxi_key_peak_y)[idx]
            peaks_y = peaks_y[:num_peaks]

            cxi_key_peak_x = self.cxi_key["peak_x"]
            peaks_x = self.cxi_handle.get(cxi_key_peak_x)[idx]
            peaks_x = peaks_x[:num_peaks]

            # [IMPROVE] Janky masking
            ## data = apply_mask(data, 1-mask)

            patch_list = get_patch_list(peaks_y, peaks_x, data, win_size = self.win_size)

        self.mpi_comm.Barrier()

        return data, peaks_y, peaks_x, patch_list
