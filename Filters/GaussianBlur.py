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
            
            
@cuda.jit
def gaussian_filter(frame, output):
    # Gaussian blur
    x, y = cuda.grid(2)  # Returns the 2D grid indices (x, y) of the current thread

    width = frame.shape[0]
    height = frame.shape[1]

    if (x < frame.shape[0] and y < frame.shape[1]):  # Check if the thread is within the bounds of the frame
        max_kernel_size = 21    # Define the maximum possible size for the Gaussian kernel in order to allocate the shared memory.
        sigma = 2.0
        kernel_size = 2 * int(4 * sigma + 0.5) + 1
        kernel = cuda.shared.array(shape=(max_kernel_size, max_kernel_size), dtype=np.float32)  # The shared memory is used to store the kernel
        compute_gaussian_kernel(kernel, kernel_size, sigma)
        m = kernel_size // 2
        n = kernel_size // 2

        # Apply the Gaussian blur to each color channel
        for c in range(3):
            sum = 0.0
            for i in range(-m, m+1):
                for j in range(-n, n+1):
                    xi = min(max(x + i, 0), width - 1)
                    yj = min(max(y + j, 0), height - 1)
                    sum += frame[xi, yj, c] * kernel[m + i, n + j]
            output[x, y, c] = sum
