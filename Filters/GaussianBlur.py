import numpy as np
import math
from numba import cuda

@cuda.jit
def compute_gaussian_kernel(d_kernel, sigma):
    x, y = cuda.grid(2)
    if x >= d_kernel.shape[0] or y >= d_kernel.shape[1]:
        return
    center_x = d_kernel.shape[0] // 2
    center_y = d_kernel.shape[1] // 2
    dx = x - center_x
    dy = y - center_y
    scale = 1.0 / (2 * math.pi * sigma ** 2)
    exponent = -(dx**2 + dy**2) / (2 * sigma ** 2)
    d_kernel[x, y] = scale * math.exp(exponent)

@cuda.jit
def convolution(oldimage, kernel, output):
    i, j = cuda.grid(2)
    if i >= output.shape[0] or j >= output.shape[1]:
        # Return if the thread is out of bounds
        return

    kh = kernel.shape[0]
    kw = kernel.shape[1]
    h_radius = kh // 2
    w_radius = kw // 2

    
    # Apply the Gaussian blur to each color channel
    channels = oldimage.shape[2]
    for c in range(channels):
        acc = 0.0
        for x in range(kh):
            for y in range(kw):
                px = i + (x - h_radius)
                py = j + (y - w_radius)
                if 0 <= px < oldimage.shape[0] and 0 <= py < oldimage.shape[1]:
                    acc += oldimage[px, py, c] * kernel[x, y]
        output[i, j, c] = acc