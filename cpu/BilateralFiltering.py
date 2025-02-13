import math
import numpy as np

def bilateral_filter(frame, sigma_s, sigma_r):
    frame = frame.astype(np.float32)    # To avoid overflow
    result = np.zeros_like(frame)
    height, width = frame.shape[0], frame.shape[1]
    kernel_size = int(2 * sigma_s + 1)
    half_kernel = kernel_size // 2

    for x in range(height):
        for y in range(width):
            total_weight = 0.0
            sum_r = 0.0
            sum_g = 0.0
            sum_b = 0.0

            center_r = frame[x, y, 0]
            center_g = frame[x, y, 1]
            center_b = frame[x, y, 2]

            for i in range(-half_kernel, half_kernel + 1):
                for j in range(-half_kernel, half_kernel + 1):
                    nx = x + i
                    ny = y + j
                    
                    if 0 <= nx < height and 0 <= ny < width:
                        current_r = frame[nx, ny, 0]
                        current_g = frame[nx, ny, 1]
                        current_b = frame[nx, ny, 2]

                        spatial_dist = i**2 + j**2
                        G_s = math.exp(-spatial_dist / (2 * sigma_s**2))

                        intensity_dist = ((current_r - center_r)**2 + 
                                        (current_g - center_g)**2 + 
                                        (current_b - center_b)**2)
                        G_r = math.exp(-intensity_dist / (2 * sigma_r**2))

                        weight = G_s * G_r
                        total_weight += weight

                        sum_r += current_r * weight
                        sum_g += current_g * weight
                        sum_b += current_b * weight

            if total_weight > 0:
                result[x, y, 0] = sum_r / total_weight
                result[x, y, 1] = sum_g / total_weight
                result[x, y, 2] = sum_b / total_weight
            else:
                result[x, y, 0] = center_r
                result[x, y, 1] = center_g
                result[x, y, 2] = center_b
                
    return result