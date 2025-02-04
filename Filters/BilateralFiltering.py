from numba import cuda
import math
import numpy as np

TILE_SIZE = 16  # Define the tile size

@cuda.jit
def gaussian_function(x_square, sigma):
    """
    Compute the Gaussian function for a given squared distance and standard deviation.
    """
    return math.exp(-x_square / (2 * sigma**2))

@cuda.jit
def bilateral_filter(frame, result, sigma_s, sigma_r):
    shared_frame = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)  # Shared memory for RGB channels
    x, y = cuda.grid(2)  # Get the current thread's absolute position in the grid
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x < frame.shape[0] and y < frame.shape[1]:  # Check if the thread is within the output image
        # Load the tile into shared memory
        for c in range(3):  # Loop over color channels
            shared_frame[tx, ty, c] = frame[x, y, c]
        
        cuda.syncthreads()
        
        kernel_size = int(sigma_s * 2 + 1)  # The kernel size should be twice the sigma space to avoid calculating negligible values and waste resources
        half_kernel_size = kernel_size // 2  # Define the half size of the kernel
        weight = 0.0  # Initialize the weight of the pixel
        result_pixel = cuda.local.array(3, dtype=np.float32)  # Initialize the output pixel to zero

        # Iterate over the kernel
        for i in range(-half_kernel_size, half_kernel_size + 1):
            for j in range(-half_kernel_size, half_kernel_size + 1):
                nx = tx + i
                ny = ty + j
                if 0 <= nx < TILE_SIZE and 0 <= ny < TILE_SIZE:  # Check if the neighboring pixel is within the shared memory bounds
                    # Compute the smoothing part:
                    G_smooth = gaussian_function(i**2 + j**2, sigma_s)  # Compute the Gaussian function for the spatial distance
                    # Compute the edge preserving part:
                    intensity_difference = 0.0
                    for c in range(3):  # Loop over color channels
                        intensity_difference += (shared_frame[tx, ty, c] - shared_frame[nx, ny, c])**2
                    G_edge = gaussian_function(intensity_difference, sigma_r)  # Compute the Gaussian function for the intensity difference
                    # Compute the weight of the neighboring pixel
                    weight += G_smooth * G_edge
                    # Compute the result
                    for c in range(3):  # Loop over color channels
                        result_pixel[c] += shared_frame[nx, ny, c] * G_smooth * G_edge
        
        # Normalize the result
        for c in range(3):  # Loop over color channels
            result[x, y, c] = result_pixel[c] / weight