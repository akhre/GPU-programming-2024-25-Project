from numba import cuda
import math
import numpy as np

@cuda.jit
def bilateral_filter(frame, result, sigma_s, sigma_r):
    x, y = cuda.grid(2)
    if x >= frame.shape[0] or y >= frame.shape[1]:
        # Outside of image bounds
        return

    kernel_size = int(2 * sigma_s + 1)
    half_kernel = kernel_size // 2

    total_weight = 0.0
    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0

    # Center pixel values
    center_r = frame[x, y, 0]
    center_g = frame[x, y, 1]
    center_b = frame[x, y, 2]

    # Iterate over kernel window
    for i in range(-half_kernel, half_kernel + 1):
        for j in range(-half_kernel, half_kernel + 1):
            nx = x + i
            ny = y + j
            
            # Boundary check
            if 0 <= nx < frame.shape[0] and 0 <= ny < frame.shape[1]:
                # Current pixel values
                current_r = frame[nx, ny, 0]
                current_g = frame[nx, ny, 1]
                current_b = frame[nx, ny, 2]

                # Spatial component
                spatial_dist = i**2 + j**2
                G_s = math.exp(-spatial_dist / (2 * sigma_s**2))

                # Range component
                intensity_dist = (current_r - center_r)**2 + \
                                 (current_g - center_g)**2 + \
                                 (current_b - center_b)**2
                G_r = math.exp(-intensity_dist / (2 * sigma_r**2))

                # Calculate weight
                weight = G_s * G_r
                total_weight += weight

                # Accumulate weighted values
                sum_r += current_r * weight
                sum_g += current_g * weight
                sum_b += current_b * weight

    # Normalize and write result
    if total_weight > 0:
        result[x, y, 0] = sum_r / total_weight
        result[x, y, 1] = sum_g / total_weight
        result[x, y, 2] = sum_b / total_weight
    else:
        result[x, y, 0] = frame[x, y, 0]
        result[x, y, 1] = frame[x, y, 1]
        result[x, y, 2] = frame[x, y, 2]