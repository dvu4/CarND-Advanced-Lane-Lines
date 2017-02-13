import glob
import numpy as np

camera_calibration_images = glob.glob('camera_cal/calibration*.jpg')

test_images = glob.glob('test_images/*.jpg')

output_images_dir = 'output_images/'

# image used to estimate the perspective transform
warp_estimate_img = test_images[3]


color_thresh_args = {
    'LUV': {
        'color': 'LUV',
        'thresh': (225, 255),
        'channel': 0
    },
    'LAB': {
        'color': 'LAB',
        'thresh': (155, 200),
        'channel': 2
    },
    'HLS': {
        'color': 'HLS',
        'thresh': (150, 255),
        'channel': 2
    }
}
x_thresh_args = {
    'orient': 'x',
    'kernel_size': 3,
    'thresh': (30, 100),
    'c_space': 'HLS',
    'channel': 2
}
y_thresh_args = {
    'orient': 'y',
    'kernel_size': 3,
    'thresh': (30, 100),
    'c_space': 'HSV',
    'channel': 1
}
mag_thresh_args = {
    'kernel_size': 5,
    'thresh': (50, 100),
    'c_space': 'gray',
    'channel': 0
}
dir_thresh_args = {
    'kernel_size': 15,
    'thresh': (0.7, 1.3),
    'c_space': 'LAB',
    'channel': 2
}
canny_thresh_args = {
    'kernel_size': 7,
    'thresh': (30, 100),
    'c_space': 'gray',
    'channel': 2
}
laplace_thresh_args = {
    'ddepth': 5,
    'kernel_size': 15,
    'thresh': (30, 100),
    'c_space': 'HLS',
    'channel': 2
}

morph_thresh_args = {
    'kernel': 5,
    'thresh': (25, 100)
}

combined_thresh_args = {
    'color_threshold': color_thresh_args,
    'abs_x_threshold': x_thresh_args,
    'abs_y_threshold': y_thresh_args,
    'mag_threshold': mag_thresh_args,
    'dir_threshold': dir_thresh_args,
    'canny_threshold': canny_thresh_args,
    'laplace_threshold': laplace_thresh_args,
    'morph_threshold': morph_thresh_args
}


comb_expr = """
combined[(thresholds['color_thresh'] == 1) |
(thresholds['x_thresh'] == 1) |
(thresholds['y_thresh'] == 1) |
(thresholds['mag_thresh'] == 1) |
(thresholds['dir_thresh'] == 1) |
(thresholds['canny_thresh'] ==1)|
(thresholds['laplace_thresh']==1) |
(thresholds['morph_thresh']==1) ] = 1
"""

hough_params = {
    'rho': 2,
    'theta': np.pi / 180,
    'threshold': 25,
    'min_line_len': 5,
    'max_line_gap': 20
}

initial_lane_params = {
    'n_bands': 9,
    'w_width': 0.2,
    'tol': 50
}