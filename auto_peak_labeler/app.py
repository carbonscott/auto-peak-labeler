import os
import numpy as np
import lmfit

from .modeling.pseudo_voigt2d import Residual


class PseudoVoigt2DLabeler:
    """
    AutoLabeler returns a binary segmentation mask (1 for peak and 0 otherwise)
    for each slice of input diffraction pattern.

    Attributes:

    """

    def __init__(self):
        return None


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
