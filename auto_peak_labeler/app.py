import sys
import os
import numpy as np
import lmfit

from mpi4py import MPI

from .modeling.pseudo_voigt2d import Residual


class PseudoVoigt2DLabeler:
    """
    AutoLabeler returns a binary segmentation mask (1 for peak and 0 otherwise)
    for each slice of input diffraction pattern.

    This code should be data source agnostic.

    Attributes:

    """
    def fit_all(self, img_peak_list):
        res_list = []
        for img_peak in img_peak_list:
            res = self.fit_one(img_peak)
            res_list.append(res)

        return res_list


    def fit_one(self, img_peak):
        """
        Return a binary segmentation mask.

        Arguments:

            img_peak : ndarray, shape [H, W], float
                       A patch that contains a peak(s).

        Returns:

            res : fitting result from lmfit
        """
        # Normalization...
        img_peak = (img_peak - img_peak.mean()) / (img_peak.std() + 1e-6)

        # Guess init values for one peak...
        H_peak, W_peak = img_peak.shape[-2:]
        bg  = img_peak.min()
        amp = img_peak.max()
        params = lmfit.Parameters()
        params.add("amp"    , value = amp),
        params.add("cy"     , value = H_peak//2),
        params.add("cx"     , value = W_peak//2),
        params.add("sigma_y", value = 2, min = 0),
        params.add("sigma_x", value = 2, min = 0),
        params.add("eta"    , value = 0.5, min = 0, max = 1),
        params.add("a"      , value = 0),
        params.add("b"      , value = 0),
        params.add("c"      , value = bg),

        # Fit the best model...
        residual = Residual(params)
        res      = residual.fit(img_peak)

        return res




class MPIPseudoVoigt2DLabeler:
    """
    This class takes a stack of images and their corresponding peak positions
    as input and then outputs model fitting results in a lmfit.result object.

    Attributes:

    """
    def __init__(self, data_source, mpi_batch_size):
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()
        self.mpi_data_tag = {
            "num_batch" : 11,
            "input"     : 12,
            "output"    : 13,
        }

        self.labeler = PseudoVoigt2DLabeler()

        self.data_source    = data_source
        self.mpi_batch_size = mpi_batch_size


    def __enter__(self):
        return self


    def __call__(self):
        return self.fit()


    def fit(self):
        mpi_comm       = self.mpi_comm
        mpi_rank       = self.mpi_rank
        mpi_size       = self.mpi_size
        mpi_data_tag   = self.mpi_data_tag
        mpi_batch_size = self.mpi_batch_size
        labeler        = self.labeler
        data_source    = self.data_source

        # ___/ MAIN \___
        mpi_comm.Barrier()
        if mpi_rank == 0:
            # Data source handles data accessing...
            patch_list = data_source.load_patch_list()

            # Inform all workers the number of batches to work on...
            batch_patch_list = np.array_split(patch_list, mpi_batch_size)
            for i in range(1, mpi_size, 1):
                num_batch = len(batch_patch_list)
                mpi_comm.send(num_batch, dest = i, tag = mpi_data_tag["num_batch"])

            # Perform model fitting for each batch...
            res_list = []
            for batch_idx, patch_list_per_batch in enumerate(batch_patch_list):
                # Split the workfload...
                patch_list_per_batch_per_chunk = np.array_split(patch_list_per_batch, mpi_size)

                for i in range(1, mpi_size, 1):
                    data_to_send = patch_list_per_batch_per_chunk[i]
                    mpi_comm.send(data_to_send, dest = i, tag = mpi_data_tag["input"])

                patch_list_current_rank = patch_list_per_batch_per_chunk[0]
                print(f"___/ Batch {batch_idx:03d} (from rank {mpi_rank:03d}) \___")
                res_list_current_rank = labeler.fit_all(patch_list_current_rank)
                res_list.extend(res_list_current_rank)

                for i in range(1, mpi_size, 1): 
                    res_list_current_rank = mpi_comm.recv(source = i, tag = mpi_data_tag["output"])
                    res_list.extend(res_list_current_rank)

        # ___/ WORKERS \___
        if mpi_rank != 0:
            num_batch = mpi_comm.recv(source = 0, tag = mpi_data_tag["num_batch"])
            for batch_idx in range(num_batch):
                patch_list_current_rank = mpi_comm.recv(source = 0, tag = mpi_data_tag["input"])
                print(f"___/ Batch {batch_idx:03d} (from rank {mpi_rank:03d}) \___")
                res_list_current_rank = labeler.fit_all(patch_list_current_rank)

                mpi_comm.send(res_list_current_rank, dest = 0, tag = mpi_data_tag["output"])

        mpi_comm.Barrier()

        return res_list if mpi_rank == 0 else None   # !!! Only worker returns None


    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The context manager achieves two things while exiting
        - Call MPI.Finalize() so users don't need to handle it.
        - Stop executing any following Python codes if it's not the main rank.

        Worker nodes will not continue with any other non-MPI Python codes.
        """
        # Considering writing saving codes here
        MPI.Finalize()

        if self.mpi_rank != 0: sys.exit(0)
