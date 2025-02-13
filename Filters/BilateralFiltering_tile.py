from numba import cuda
import math
import numpy as np

TILE_SIZE = 16  # Size of the shared memory tile

@cuda.jit
def bilateral_filter(frame, result, sigma_s, sigma_r):
    # Shared memory for caching image tiles
    shared_frame = cuda.shared.array((TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)
    
    # Thread and block indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    
    # Pixel coordinates in the output image
    x = bx * TILE_SIZE + tx
    y = by * TILE_SIZE + ty
    
    # Load the tile into shared memory
    if x < frame.shape[0] and y < frame.shape[1]:
        for c in range(3):  # Load all color channels
            shared_frame[tx, ty, c] = frame[x, y, c]
    else:
        for c in range(3):
            shared_frame[tx, ty, c] = 0.0  # Pad with zeros if out of bounds
    
    # Wait for all threads to finish loading
    cuda.syncthreads()
    
    # Only threads within the image bounds compute the result
    if x < frame.shape[0] and y < frame.shape[1]:
        # Initialize accumulators
        total_weight = 0.0
        sum_r = 0.0
        sum_g = 0.0
        sum_b = 0.0
        
        # Center pixel values
        center_r = shared_frame[tx, ty, 0]
        center_g = shared_frame[tx, ty, 1]
        center_b = shared_frame[tx, ty, 2]
        
        # Kernel radius
        kernel_size = int(2 * sigma_s + 1)
        half_kernel = kernel_size // 2
        
        # Iterate over the kernel window
        for i in range(-half_kernel, half_kernel + 1):
            for j in range(-half_kernel, half_kernel + 1):
                # Neighbor coordinates in shared memory
                nx = tx + i
                ny = ty + j
                
                # Check if the neighbor is within the shared memory tile
                if 0 <= nx < TILE_SIZE and 0 <= ny < TILE_SIZE:
                    # Neighbor pixel values
                    current_r = shared_frame[nx, ny, 0]
                    current_g = shared_frame[nx, ny, 1]
                    current_b = shared_frame[nx, ny, 2]
                    
                    # Spatial distance
                    spatial_dist = i**2 + j**2
                    G_s = math.exp(-spatial_dist / (2 * sigma_s**2))
                    
                    # Intensity distance
                    intensity_dist = (current_r - center_r)**2 + \
                                     (current_g - center_g)**2 + \
                                     (current_b - center_b)**2
                    G_r = math.exp(-intensity_dist / (2 * sigma_r**2))
                    
                    # Weight
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