import numpy as np
import lmfit

from math import log


class PseudoVoigt2D:
    """
    PseudoVoigt2D produces a pseudo Voigt 2D profile.

    Attributes:
        params : lmfit.Parameters
            Example: { "amp" : {"value" : 0, "min" : 0,} }
    """
    def __init__(self, params, includes_bg = True):
        self.params      = params
        self.includes_bg = includes_bg


    def _gaussian_2d(self, y, x, cy, cx, sigma_y, sigma_x):
        return np.exp(-(((y-cy)**2)/(2*sigma_y**2) + ((x-cx)**2)/(2*sigma_x**2)))


    def _lorentzian_2d(self, y, x, cy, cx, gamma_y, gamma_x):
        return 1/(1 + ((y-cy)**2)/(gamma_y**2) + ((x-cx)**2)/(gamma_x**2))


    def _pseudo_voigt2D(self, y, x, amp, cy, cx, sigma_y, sigma_x, gamma_y, gamma_x, eta):
        ## gamma_y = sigma_y * 2 * log(2)
        ## gamma_x = sigma_x * 2 * log(2)
        res = (1-eta)*self._gaussian_2d(y, x, cy, cx, sigma_y, sigma_x) + \
              eta    *self._lorentzian_2d(y, x, cy, cx, gamma_y, gamma_x)
        return amp * res


    def _plane2d(self, y, x, a, b, c):
        """
        Return a 2D plane profile.

        Arguments:

        Returns:

            a 2D plane profile : scalar or ndarray
        """
        return a * y + b * x + c


    def __call__(self, y, x):
        params  = self.params

        amp     = params["amp"]
        cy      = params["cy"]
        cx      = params["cx"]
        sigma_y = params["sigma_y"]
        sigma_x = params["sigma_x"]
        gamma_y = params["gamma_y"]
        gamma_x = params["gamma_x"]
        eta     = params["eta"]
        a       = params["a"]
        b       = params["b"]
        c       = params["c"]

        model_profile = self._pseudo_voigt2D(y, x, amp, cy, cx, sigma_y, sigma_x, gamma_y, gamma_x, eta)

        if self.includes_bg:
            bg = self._plane2d(y, x, a, b, c)
            model_profile += bg

        return model_profile


    def update_params(self, params):
        self.params.update(params)


    def fwhm_gaussian(self, sigma):
        """Calculate the FWHM of the Gaussian component."""
        return 2 * sqrt(2 * log(2)) * sigma


    def fwhm_lorentzian(self, gamma):
        """Calculate the FWHM of the Lorentzian component."""
        return 2 * gamma


    def fwhm_pseudovoigt(self, sigma, gamma, eta):
        """Approximate the FWHM of the Pseudo-Voigt profile."""
        fwhm_gaussian = self.fwhm_gaussian(sigma)
        fwhm_lorentzian = self.fwhm_lorentzian(gamma)
        return eta * fwhm_lorentzian + (1 - eta) * fwhm_gaussian




class Residual:
    """
    Residual calculates the residual values between model and data.

    Attributes:

        params : lmfit.Parameters
            Example: { "amp" : {"value" : 0, "min" : 0,} }

        fitting_method : 'leastsq' or 'least_squares'
            More info: https://lmfit.github.io/lmfit-py/fitting.html

    """
    def __init__(self, params):
        self.params         = params
        self.model          = PseudoVoigt2D(params)
        self.fitting_method = 'leastsq'

        return None


    def _residual(self, params, data, **kwargs):
        model = self.model

        H, W = data.shape[-2:]
        y    = np.arange(0, H)
        x    = np.arange(0, W)
        Y, X = np.meshgrid(y, x, indexing = 'ij')

        ## model.update_params(params)
        model.params.update(params)
        model_eval = model(Y, X)

        return model_eval - data


    def fit(self, data, **kwargs):
        method = self.fitting_method
        res    = lmfit.minimize(self._residual,
                                self.params,
                                method     = method,
                                max_nfev   = 2000,
                                nan_policy = 'omit',
                                args       = (data, ),
                                **kwargs)

        return res
