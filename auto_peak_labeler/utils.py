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

        # ...Padding
        x_pad_l = -min(x - win_size, 0)            # ...Lower
        x_pad_u =  max(x + win_size + 1 - W, 0)    # ...Upper
        y_pad_l = -min(y - win_size, 0)            # ...Lower
        y_pad_u =  max(y + win_size + 1 - H, 0)    # ...Upper
        padding = ((y_pad_l, y_pad_u), (x_pad_l, x_pad_u))
        img_peak = np.pad(img_peak, padding, mode='constant', constant_values=0)

        patch_list.append(img_peak)

    return patch_list




def apply_mask(data, mask, mask_value = np.nan):
    """ 
    Return masked data.

    Args:
        data: numpy.ndarray with the shape of (B, H, W).·
              - B: batch of images.
              - H: height of an image.
              - W: width of an image.

        mask: numpy.ndarray with the shape of (B, H, W).·

    Returns:
        data_masked: numpy.ndarray.
    """ 
    # Mask unwanted pixels with np.nan...
    data_masked = np.where(mask, data, mask_value)

    return data_masked
