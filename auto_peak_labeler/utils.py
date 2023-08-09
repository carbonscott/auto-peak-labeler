import os
import numpy as np


def get_patch_list(peaks_y, peaks_x, img, win_size):
    patch_list = []
    for enum_peak_idx, (peak_y, peak_x) in enumerate(zip(peaks_y, peaks_x)):
        # Obtain the peak area with a certain window size...
        # ...Get the rough location of the peak
        y, x = round(peak_y), round(peak_x)

        # ...Define the area
        H, W = img.shape[-2:]
        x_min = max(x - win_size    , 0)
        x_max = min(x + win_size + 1, W)    # offset by 1 to allow including the rightmost index
        y_min = max(y - win_size    , 0)
        y_max = min(y + win_size + 1, H)

        # ...Crop
        # Both variables are views of the original data
        img_peak = img[y_min:y_max, x_min:x_max]

        patch_list.append(img_peak)

    return patch_list
