import os
import sys
import signal
import ray
import numpy as np

from scipy.ndimage import center_of_mass

from .modeling.pseudo_voigt2d import Residual


class PeakFitter:
    def __init__(self, num_cpus = 10):
        self.num_cpus = num_cpus


    def fit_all_images(self, image_list):
        # Shutdown ray clients during a Ctrl+C event...
        def signal_handler(sig, frame):
            if ray.is_initialized():
                print('SIGINT (Ctrl+C) caught, shutting down Ray...')
                ray.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Init ray...
        # Check if RAY_ADDRESS is set in the environment
        USES_MULTI_NODES = os.getenv('USES_MULTI_NODES')
        if USES_MULTI_NODES:
            ray.init(address = 'auto')
        else:
            ray.init(num_cpus = self.num_cpus)

        def initial_guess_for_peak(peak_image):
            # Estimate centroid (cy, cx)
            cy, cx = center_of_mass(peak_image)

            # Estimate amplitude (amp)
            amp = np.max(peak_image)

            # Estimate width parameters (sigma_x, sigma_y)
            # Assuming a symmetric peak for initial guess
            y, x = np.indices(peak_image.shape)
            weighted_distances = peak_image * ((x - cx)**2 + (y - cy)**2)
            sigma = np.sqrt(np.sum(weighted_distances) / np.sum(peak_image))

            # Estimate background (c)
            # Assuming the background is the average of the corners of the image
            corners = [peak_image[0,0], peak_image[0,-1], peak_image[-1,0], peak_image[-1,-1]]
            c = np.mean(corners)

            # Balance between Gaussian and Lorentzian (eta)
            eta = 0.5

            # For gamma, start with the same value as sigma
            gamma = sigma

            return {
                'amp'    : amp,
                'cy'     : cy,
                'cx'     : cx,
                'sigma_y': sigma,
                'sigma_x': sigma,
                'gamma_y': gamma,
                'gamma_x': gamma,
                'eta'    : eta,
                'a'      : 0,
                'b'      : 0,
                'c'      : c
            }

        def perform_fitting(image, initial_params):
            residual = Residual(initial_params)
            result = residual.fit(image)
            return result

        @ray.remote
        def fit_single_image(image):
            initial_params = initial_guess_for_peak(image)
            return perform_fitting(image, initial_params)


        futures = [fit_single_image.remote(image) for image in image_list]
        results = ray.get(futures)
        return results

    def shutdown(self):
        ray.shutdown()
