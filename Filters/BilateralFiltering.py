from numba import cuda
import math
import numpy as np

TILE_SIZE = 96  # Define the tile size

@cuda.jit
def gaussian_function(x_square, sigma):
    """
    Compute the Gaussian function for a given squared distance and standard deviation.
    """
    return math.exp(-x_square / (2 * sigma**2)) # / (2 * math.pi * sigma**2)  


@cuda.jit
def bilateral_filter_x_channel(frame, result, sigma_s, sigma_r):
    shared_frame = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=np.float32)  # Adjust the shape based on your block size
    x, y = cuda.grid(2)  # Get the current thread's absolute position in the grid
    tx = cuda.threadIdx.x + 1
    ty = cuda.threadIdx.y + 1
    
    if x < frame.shape[0] and y < frame.shape[1]:  # Check if the thread is within the output image
        # Load the tile into shared memory
        shared_frame[tx, ty] = frame[x, y]
        
        # Load halo elements into shared memory
        if tx == 1 and x > 0:
            shared_frame[0, ty] = frame[x - 1, y]
        if tx == TILE_SIZE and x < frame.shape[0] - 1:
            shared_frame[TILE_SIZE + 1, ty] = frame[x + 1, y]
        if ty == 1 and y > 0:
            shared_frame[tx, 0] = frame[x, y - 1]
        if ty == TILE_SIZE and y < frame.shape[1] - 1:
            shared_frame[tx, TILE_SIZE + 1] = frame[x, y + 1]
        
        # Load corner elements into shared memory
        if tx == 1 and ty == 1 and x > 0 and y > 0:
            shared_frame[0, 0] = frame[x - 1, y - 1]
        if tx == 1 and ty == TILE_SIZE and x > 0 and y < frame.shape[1] - 1:
            shared_frame[0, TILE_SIZE + 1] = frame[x - 1, y + 1]
        if tx == TILE_SIZE and ty == 1 and x < frame.shape[0] - 1 and y > 0:
            shared_frame[TILE_SIZE + 1, 0] = frame[x + 1, y - 1]
        if tx == TILE_SIZE and ty == TILE_SIZE and x < frame.shape[0] - 1 and y < frame.shape[1] - 1:
            shared_frame[TILE_SIZE + 1, TILE_SIZE + 1] = frame[x + 1, y + 1]
       
        
        cuda.syncthreads()
        
        kernel_size = sigma_s * 2 + 1  # The kernel size should be twice the sigma space to avoid calculating negligible values and waste resources
        half_kernel_size = kernel_size // 2  # Define the half size of the kernel
        weight = 0.0  # Initialize the weight of the pixel
        result[x, y] = 0.0  # Initialize the output image to zero

        # Iterate over the kernel
        for i in range(-half_kernel_size, half_kernel_size + 1):
            for j in range(-half_kernel_size, half_kernel_size + 1):
                if tx + i >= 0 and tx + i < TILE_SIZE + 2 and ty + j >= 0 and ty + j < TILE_SIZE + 2:  # Check if the neighboring pixel is within the shared memory bounds
                    # Compute the smoothing part:
                    G_smooth = gaussian_function(i**2 + j**2, sigma_s)  # Compute the Gaussian function for the spatial distance
                    # Compute the edge preserving part:
                    intensity_difference_image = shared_frame[tx, ty] - shared_frame[tx + i, ty + j]  # Compute the intensity difference between the neighboring pixel and the current pixel
                    G_edge = gaussian_function(intensity_difference_image**2, sigma_r)  # Compute the Gaussian function for the intensity difference
                    # Compute the weight of the neighboring pixel
                    weight += G_smooth * G_edge
                    # Compute the result
                    result[x, y] += shared_frame[tx + i, ty + j] * G_smooth * G_edge
                    
        # Normalize the result
        result[x, y] /= weight