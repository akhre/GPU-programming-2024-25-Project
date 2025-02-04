#!/usr/bin/python3

import sys
import cv2
import numpy as np
from numba import cuda
import VideoFrame
from Filters.BicubicInterpolation import bicubic_interpolation
# from Filters.BilateralFiltering_v2 import bilateral_filter_x_channel  # There is a problem with the shared memorry, the computation happens (as expected) only on a little tile, but this cause glitches on the output
from Filters.BilateralFiltering import bilateral_filter_x_channel
from Filters.GaussianBlur import compute_gaussian_kernel
import math
import logging

# VIDEO_PATH = "/media/diego/Volume/Polito/Magistrale/GPU programming/Lab/Project/videos/videoplayback.mp4"
VIDEO_PATH = "/media/diego/Volume/Polito/Magistrale/GPU programming/Lab/Project/videos/KoNViD_1k_videos/12099271596.mp4"

# Configure logging for each filter
logging.basicConfig(level=logging.ERROR)
# filter1_logger = logging.getLogger("Filter1")
# filter2_logger = logging.getLogger("Filter2")
# filter3_logger = logging.getLogger("Filter3")
# filter4_logger = logging.getLogger("Filter4")


# filter1_handler = logging.FileHandler("filter1.log", "w")
# filter2_handler = logging.FileHandler("filter2.log", "w")
# filter3_handler = logging.FileHandler("filter3.log", "w")
# filter4_handler = logging.FileHandler("filter4.log", "w")

# filter1_logger.addHandler(filter1_handler)
# filter2_logger.addHandler(filter2_handler)
# filter3_logger.addHandler(filter3_handler)
# filter4_logger.addHandler(filter4_handler)


@cuda.jit
def process_frame_gpu_filtro_1(frame, output):
    x, y = cuda.grid(2)  # Returns the 2D grid indices (x, y) of the current thread
    if ( x < frame.shape[0] and y < frame.shape[1]):  # Check if the thread is within the bounds of the frame
        output[x, y, 0] = frame[x, y, 0]
        output[x, y, 1] = 0
        output[x, y, 2] = 0


@cuda.jit
def process_frame_gpu_filtro_2(frame, output):
    # Gaussian blur
    x, y = cuda.grid(2)  # Returns the 2D grid indices (x, y) of the current thread

    width = frame.shape[0]
    height = frame.shape[1]

    if (x < frame.shape[0] and y < frame.shape[1]):  # Check if the thread is within the bounds of the frame
        # Define the 13x13 Gaussian kernel
        kernel_size = 13
        sigma = 2.0
        kernel = cuda.shared.array(shape=(kernel_size, kernel_size), dtype=np.float32)  # The shared memory is used to store the kernel
        compute_gaussian_kernel(kernel, kernel_size, sigma)

        # Apply the Gaussian blur to each color channel
        for c in range(3):
            sum = 0.0
            for i in range(-6, 7):
                for j in range(-6, 7):
                    xi = min(max(x + i, 0), width - 1)
                    yj = min(max(y + j, 0), height - 1)
                    sum += kernel[i + 6, j + 6] * frame[xi, yj, c]

            output[x, y, c] = sum


@cuda.jit
def process_frame_gpu_filtro_3(frame, output):
    x, y = cuda.grid(2)  # Returns the 2D grid indices (x, y) of the current thread
    if (x < frame.shape[0] and y < frame.shape[1]):  # Check if the thread is within the bounds of the frame
        output[x, y] = bicubic_interpolation(frame[x, y])


def process_frame_gpu_filtro_4(frame, sigma_s, sigma_r):
    # Bilateral filtering
    # Split the frame into RGB channels
    frame_r = np.ascontiguousarray(frame[:, :, 0])
    frame_g = np.ascontiguousarray(frame[:, :, 1])
    frame_b = np.ascontiguousarray(frame[:, :, 2])
    
    # Allocate pinned (page-locked) arrays for faster transfer
    frame_r_pinned = cuda.pinned_array(frame_r.shape, dtype=frame_r.dtype)
    frame_g_pinned = cuda.pinned_array(frame_g.shape, dtype=frame_g.dtype)
    frame_b_pinned = cuda.pinned_array(frame_b.shape, dtype=frame_b.dtype)
    
    # Copy the host data into pinned arrays
    frame_r_pinned[:] = frame_r[:]
    frame_g_pinned[:] = frame_g[:]
    frame_b_pinned[:] = frame_b[:]
    
    
    # Create CUDA streams
    stream1 = cuda.stream()
    stream2 = cuda.stream()
    stream3 = cuda.stream()
    
    # Transfer pinned data to device asynchronously using separate streams
    frame_r_device = cuda.to_device(frame_r_pinned, stream=stream1)
    frame_g_device = cuda.to_device(frame_g_pinned, stream=stream2)
    frame_b_device = cuda.to_device(frame_b_pinned, stream=stream3)
    
    
    frame_shape = (frame.shape[0], frame.shape[1])
    # Allocate memory for result on the device
    result_r_device = cuda.device_array(frame_shape, dtype=np.float32, stream=stream1)
    result_g_device = cuda.device_array(frame_shape, dtype=np.float32, stream=stream2)
    result_b_device = cuda.device_array(frame_shape, dtype=np.float32, stream=stream3)
    result_device = cuda.device_array((frame.shape[0], frame.shape[1], 3), dtype=np.float32)


    # Define the grid and block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(frame_shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(frame_shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Launch the kernel for each channel
    bilateral_filter_x_channel[blocks_per_grid, threads_per_block, stream1](frame_r_device, result_r_device, sigma_s, sigma_r)
    bilateral_filter_x_channel[blocks_per_grid, threads_per_block, stream2](frame_g_device, result_g_device, sigma_s, sigma_r)
    bilateral_filter_x_channel[blocks_per_grid, threads_per_block, stream3](frame_b_device, result_b_device, sigma_s, sigma_r)
    
    # Synchronize the streams 
    # stream1.synchronize()
    # stream2.synchronize()
    # stream3.synchronize()
    cuda.synchronize()
    
    # Combine the channels back into an image on the GPU
    combine_channels[blocks_per_grid, threads_per_block](result_r_device, result_g_device, result_b_device, result_device)

    # Copy the final result back to the host
    result = result_device.copy_to_host()
    
    # Convert the result to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)   # Clip may be not necessary
    return result



@cuda.jit
def combine_channels(result_r, result_g, result_b, result):
    x, y = cuda.grid(2)
    if x < result.shape[0] and y < result.shape[1]:
        result[x, y, 0] = result_r[x, y]
        result[x, y, 1] = result_g[x, y]
        result[x, y, 2] = result_b[x, y]





def measure_distortion(original_frame, filtered_frame):
    # Convert frames to grayscale
    original_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    filtered_gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)

    # Compute the Mean Squared Error (MSE) between the original and filtered frames
    mse = np.mean((original_gray - filtered_gray) ** 2)

    # Compute the Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))

    return mse, psnr


class OriginalFrame(VideoFrame.VideoFrame):
    def __init__(self):
        super().__init__(None)  # Call the parent class constructor without a parent
        self.screens = []
        self.vid = None
    
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            for screen in self.screens:
                screen.update(frame)
        else:
            self.vid.release()
            return False  # Indicate that the video has ended
        return True  # Indicate that the video is still playing
    
    def load_video(self, video):
        self.vid = cv2.VideoCapture(video)  # Open the video file
        
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class Filter1Frame(VideoFrame.VideoFrame):
    def apply_filter(self, frame):
        # Allocate memory on the GPU
        d_frame = cuda.to_device(np.ascontiguousarray(frame))
        # d_frame = cuda.to_device(frame)
        d_output = cuda.device_array_like(frame)
        
        # Define the grid and block dimensions
        threads_per_block = (16, 16)
        
        blocks_per_grid_x = int(np.ceil((frame.shape[0] + threads_per_block[0] - 1) / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil((frame.shape[1] + threads_per_block[1] - 1) / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        process_frame_gpu_filtro_1[blocks_per_grid, threads_per_block](d_frame, d_output)
        filtered_frame = d_output.copy_to_host()
        mse, psnr = measure_distortion(frame, filtered_frame)
        print(f"Filter 1 - MSE: {mse}, PSNR: {psnr}")
        # filter1_logger.info(f"MSE: {mse}, PSNR: {psnr}")
        return filtered_frame
        
               
class Filter2Frame(VideoFrame.VideoFrame):
    def apply_filter(self, frame):
        # Allocate memory on the GPU
        d_frame = cuda.to_device(np.ascontiguousarray(frame))
        # d_frame = cuda.to_device(frame)
        d_output = cuda.device_array_like(frame)

        # Define the grid and block dimensions
        threads_per_block = (16, 16)
        
        blocks_per_grid_x = int(np.ceil((frame.shape[0] + threads_per_block[0] - 1) / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil((frame.shape[1] + threads_per_block[1] - 1) / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        process_frame_gpu_filtro_2[blocks_per_grid, threads_per_block](d_frame, d_output)
        # bilateral_filter[blocks_per_grid, threads_per_block](d_frame, d_output, 15.0, 0.1)  # sigma_s = 15.0, sigma_r = 0.1
        filtered_frame = d_output.copy_to_host()
        mse, psnr = measure_distortion(frame, filtered_frame)
        print(f"Filter 2 - MSE: {mse}, PSNR: {psnr}")
        # filter2_logger.info(f"MSE: {mse}, PSNR: {psnr}")
        return filtered_frame


class Filter3Frame(VideoFrame.VideoFrame):

    def apply_filter(self, frame):
        # Scale factor for the bicubic interpolation
        scale = 2.0
        # scale = 0.5

        # Allocate memory on the GPU
        d_frame = cuda.to_device(np.ascontiguousarray(frame))
        # d_frame = cuda.to_device(frame)
        d_output = cuda.device_array_like(frame)

        # Define the grid and block dimensions
        threads_per_block = (16, 16)
        
        blocks_per_grid_x = int(np.ceil((frame.shape[0] + threads_per_block[0] - 1) / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil((frame.shape[1] + threads_per_block[1] - 1) / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        bicubic_interpolation[blocks_per_grid, threads_per_block](d_frame, d_output, scale)
        filtered_frame = d_output.copy_to_host()
        mse, psnr = measure_distortion(frame, filtered_frame)
        print(f"Filter 3 - MSE: {mse}, PSNR: {psnr}")
        # filter3_logger.info(f"MSE: {mse}, PSNR: {psnr}")
        return filtered_frame


class Filter4Frame(VideoFrame.VideoFrame):

    def apply_filter(self, frame):
        # Apply the bilateral filter
        filtered_frame = process_frame_gpu_filtro_4(frame, 8, 0.1)  # sigma_s = 15.0, sigma_r = 0.1
        # filtered_frame = frame
        # Measure distortion
        mse, psnr = measure_distortion(frame, filtered_frame)
        print(f"Filter 4 - MSE: {mse}, PSNR: {psnr}")

        # Log the results
        # filter4_logger.info(f"MSE: {mse}, PSNR: {psnr}")

        return filtered_frame
    
        
def main():
    camera_frame = OriginalFrame()
    filter_1_frame = Filter1Frame(None)
    filter_2_frame = Filter2Frame(None)
    filter_3_frame = Filter3Frame(None)
    filter_4_frame = Filter4Frame(None)

    camera_frame.screens.append(filter_1_frame)
    camera_frame.screens.append(filter_2_frame)
    camera_frame.screens.append(filter_3_frame)
    camera_frame.screens.append(filter_4_frame)

    camera_frame.load_video(VIDEO_PATH)

    while camera_frame.update():
        pass  # Continue updating until the video ends

    return 0

if __name__ == "__main__":
    sys.exit(main())