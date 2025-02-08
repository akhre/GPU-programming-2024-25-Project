import numpy as np
from numba import cuda

FILTER_WIDTH = 3
FILTER_HEIGHT = 3

@cuda.jit
def laplacianFilter(frame, result, width, height):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    # Define the kernel to use
    kernel = cuda.local.array((3, 3), dtype=np.float32)
    
    # Define the kernel values: Positive laplacian mask
    kernel[0, 0] = 0
    kernel[0, 1] = 1
    kernel[0, 2] = 0
    kernel[1, 0] = 1
    kernel[1, 1] = -4
    kernel[1, 2] = 1
    kernel[2, 0] = 0
    kernel[2, 1] = 1
    kernel[2, 2] = 0
    
    # Define the kernel values: Negative laplacian mask
    # kernel[0, 0] = 0
    # kernel[0, 1] = -1
    # kernel[0, 2] = 0
    # kernel[1, 0] = -1
    # kernel[1, 1] = 4
    # kernel[1, 2] = -1
    # kernel[2, 0] = 0
    # kernel[2, 1] = -1
    # kernel[2, 2] = 0
    
    
    # Check if inside the limit of the frame:
    if((x >= FILTER_WIDTH // 2) and (x < (width - FILTER_WIDTH // 2)) and (y >= FILTER_HEIGHT // 2) and (y < (height - FILTER_HEIGHT // 2))):
        # Initialize the result
        sum : float = 0
        for kx in range(-FILTER_WIDTH // 2, FILTER_WIDTH // 2 + 1):
            for ky in range(-FILTER_HEIGHT // 2, FILTER_HEIGHT // 2 + 1):
                fl : float = frame[x + kx][y + ky]
                sum += fl * kernel[kx + FILTER_WIDTH // 2][ky + FILTER_HEIGHT // 2]
                
        result[x][y] = sum