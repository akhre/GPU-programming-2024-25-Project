import numpy as np
import math
from numba import cuda

@cuda.jit
def compute_gaussian_kernel(kernel, kernel_size, sigma):
    sum_val = 0.0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x_dist = i - (kernel_size // 2)
            y_dist = j - (kernel_size // 2)
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * math.exp(-(x_dist**2 + y_dist**2) / (2 * sigma**2))
            sum_val += kernel[i, j]
    
    # Normalize the kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] /= sum_val