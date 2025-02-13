#!/usr/bin/python3

import sys
import cv2
import numpy as np
from numba import cuda
import VideoFrameOptimized as VideoFrame
from Filters.BicubicInterpolation_optimized import bicubic_interpolation
from Filters.BilateralFiltering_tile import bilateral_filter
from Filters.GaussianBlur import compute_gaussian_kernel, convolution
from Filters.LaplacianFilter import laplacianFilter
import math
import logging
import os # Used to get the video name
from datetime import datetime # Used in the logs to timestamp the execution

RESULTS_DIR = "results/gpu_optimized" # Directory to store the results
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)    # Create the results directory if it does not exist

VIDEO_PATH = ''
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    print("Usage: python main_profiler.py <video_path>")
    sys.exit(1)

# Configure logging for each filter
logging.basicConfig(level=logging.INFO)
filter1_logger = logging.getLogger("Filter1")
filter2_logger = logging.getLogger("Filter2")
filter3_logger = logging.getLogger("Filter3")
filter4_logger = logging.getLogger("Filter4")


filter1_handler = logging.FileHandler(f"{RESULTS_DIR}/{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_filter_laplacian.log", "w")
filter2_handler = logging.FileHandler(f"{RESULTS_DIR}/{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_filter_gaussian_blur.log", "w")
filter3_handler = logging.FileHandler(f"{RESULTS_DIR}/{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_filter_bicubic_interpolation.log", "w")
filter4_handler = logging.FileHandler(f"{RESULTS_DIR}/{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_filter_bilateral.log", "w")

filter1_logger.addHandler(filter1_handler)
filter2_logger.addHandler(filter2_handler)
filter3_logger.addHandler(filter3_handler)
filter4_logger.addHandler(filter4_handler)


def measure_distortion(original_frame, filtered_frame):
    
    if (original_frame.shape[0] != filtered_frame.shape[0]) or (original_frame.shape[1] != filtered_frame.shape[1]):
        # Resize the original frame to match the dimensions of the filtered frame
        original_frame = cv2.resize(original_frame, (filtered_frame.shape[1], filtered_frame.shape[0]))

    
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
        
        self.d_output_filter_1 = None
        self.d_output_filter_2 = None
        self.d_output_filter_3 = None
        self.d_output_filter_4 = None

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.apply_filters(frame)
        else:
            self.vid.release()
            return False  # Indicate that the video has ended
        return True  # Indicate that the video is still playing
    
    def load_video(self, video):
        self.vid = cv2.VideoCapture(video)  # Open the video file
        self.update()
    def apply_filters(self, frame):
        # Create the CUDA streams to execute the filters in parallel
        stream_copy = cuda.stream()
        stream_filter_1 = cuda.stream()
        stream_filter_2 = cuda.stream()
        stream_filter_3 = cuda.stream()
        stream_filter_4 = cuda.stream()
        
        # Copy the data to the GPU (with the streams so asynchronous copies can be performed)
        d_frame = cuda.to_device(frame, stream_copy)
        
        if self.d_output_filter_1 is None:
            self.d_output_filter_1 = cuda.device_array_like(frame, stream_copy)
        if self.d_output_filter_2 is None:
            self.d_output_filter_2 = cuda.device_array_like(frame, stream_copy)
        if self.d_output_filter_3 is None:
            self.d_output_filter_3 = cuda.device_array_like(frame, stream_copy)
        if self.d_output_filter_4 is None:
            self.d_output_filter_4 = cuda.device_array((frame.shape[0], frame.shape[1], 3), dtype=np.float32, stream=stream_copy)
            
        stream_copy.synchronize()
        
        # Apply the filters:
        filtered_frame1 = filter_1_frame.apply_filter(frame, d_frame, self.d_output_filter_1, stream_filter_1)
        filtered_frame2 = filter_2_frame.apply_filter(frame, d_frame, self.d_output_filter_2, stream_filter_2)
        filtered_frame3 = filter_3_frame.apply_filter(frame, d_frame, self.d_output_filter_3, stream_filter_3)
        filtered_frame4 = filter_4_frame.apply_filter(frame, d_frame, self.d_output_filter_4, stream_filter_4)
        

        # Update the frames
        stream_filter_1.synchronize()
        filter_1_frame.update(filtered_frame1)
        
        stream_filter_2.synchronize()
        filter_2_frame.update(filtered_frame2)
        
        stream_filter_3.synchronize()
        filter_3_frame.update(filtered_frame3)
        
        stream_filter_4.synchronize()
        filter_4_frame.update(filtered_frame4)
        

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class Filter1Frame(VideoFrame.VideoFrame):
    def apply_filter(self, frame, d_frame, d_output, stream):
        """
        Filter: Laplacian Filter
        """
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Allocate the new frame on the GPU memory:
        d_frame_gray = cuda.to_device(np.ascontiguousarray(gray_frame))
        # d_frame = cuda.to_device(frame)
        
        
        # Define the grid and block dimensions
        threads_per_block = (16, 16)
        
        # Compute the grid size to cover the whole frame:
        blocks_per_grid_x = int(
            np.ceil((frame.shape[0] + threads_per_block[0] - 1) / threads_per_block[0])
        )
        blocks_per_grid_y = int(
            np.ceil((frame.shape[1] + threads_per_block[1] - 1) / threads_per_block[1])
        )
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Create CUDA events for timing
        start_event = cuda.event()
        end_event = cuda.event()

        # Record the start event
        start_event.record()

        # Launch the kernel
        laplacianFilter[blocks_per_grid, threads_per_block, stream](
            d_frame_gray, d_output, frame.shape[0], frame.shape[1]
        )
        
        cuda.synchronize()  # Wait for all threads to complete
        # Record the end event
        end_event.record()

        # Wait for the end event to complete
        end_event.synchronize()

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = cuda.event_elapsed_time(start_event, end_event)

        # Copy the result back to the host
        filtered_frame = d_output.copy_to_host()

        # Log the results
        filter1_logger.info(f"Timestamp: {datetime.now()}, EXECUTION TIME ms: {elapsed_time_ms}")

        return filtered_frame


class Filter2Frame(VideoFrame.VideoFrame):
    def apply_filter(self, frame, d_frame, d_output, stream):
        """
        Filter: Gaussian Blur
        """
        
        # Determine the filter size
        sigma = 2.0
        filter_size = 2 * int(4 * sigma + 0.5) + 1
        kernel = np.zeros((filter_size, filter_size), dtype=np.float32)
        
        # Allocate memory on the GPU
        d_kernel = cuda.to_device(kernel)
        
        # Define the grid and block dimensions
        threads_per_block = (16, 16)
        blocks_per_grid_x = (filter_size + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (filter_size + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        # Create CUDA events for timing
        start_event = cuda.event()
        end_event = cuda.event()

        # Record the start event
        start_event.record()
        
        # Compute Gaussian kernel on the device
        compute_gaussian_kernel[blocks_per_grid, threads_per_block, stream](d_kernel, sigma)
        
        # Normalize the kernel (copy back to host, sum, normalize, send back)
        kernel = d_kernel.copy_to_host()
        kernel_sum = kernel.sum()
        kernel /= kernel_sum
        d_kernel = cuda.to_device(kernel)
        

        
        # Configure convolution kernel launch parameters
        # threads_per_block = (16, 16)
        grid_x = int(np.ceil((frame.shape[0] + threads_per_block[0] - 1) / threads_per_block[0]))
        grid_y = int(np.ceil((frame.shape[1] + threads_per_block[1] - 1) / threads_per_block[1]))
        blocks_per_grid = (grid_x, grid_y)
        # Perform convolution
        convolution[blocks_per_grid, threads_per_block](frame, d_kernel, d_output)
        

        # Record the end event
        end_event.record()

        # Wait for the end event to complete
        end_event.synchronize()

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = cuda.event_elapsed_time(start_event, end_event)
        
        filtered_frame = d_output.copy_to_host()
        filter2_logger.info(f"Timestamp: {datetime.now()}, EXECUTION TIME ms: {elapsed_time_ms}")
        return filtered_frame


class Filter3Frame(VideoFrame.VideoFrame):

    def apply_filter(self, frame, d_frame, d_output, stream):
        """
        Filter: Bicubic interpolation
        """
        # Scale factor for the bicubic interpolation
        scale = 2.0

        # Calculate the dimensions of the scaled frame
        scaled_height = int(frame.shape[0] * scale)
        scaled_width = int(frame.shape[1] * scale)

        # Allocate memory for the scaled frame on the GPU
        d_output = cuda.device_array((scaled_height, scaled_width, frame.shape[2]), dtype=frame.dtype)

        # Define the grid and block dimensions
        threads_per_block = (16, 16)

        blocks_per_grid_x = int(np.ceil((scaled_height + threads_per_block[0] - 1) / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil((scaled_width + threads_per_block[1] - 1) / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        # Create CUDA events for timing
        start_event = cuda.event()
        end_event = cuda.event()

        # Record the start event
        start_event.record()

        # Launch the kernel
        bicubic_interpolation[blocks_per_grid, threads_per_block, stream](d_frame, d_output, scale)
        
        cuda.synchronize()  # wait for all threads to complete. The copy to the host performs an implicit synchronization, so the call to cuda.syncronize is not really necessary.
        
        # Record the end event
        end_event.record()

        # Wait for the end event to complete
        end_event.synchronize()

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = cuda.event_elapsed_time(start_event, end_event)
        
        filtered_frame = d_output.copy_to_host()
        mse, psnr = measure_distortion(frame, filtered_frame)
        
        filter3_logger.info(f"Timestamp: {datetime.now()}, EXECUTION TIME ms: {elapsed_time_ms}, MSE: {mse}, PSNR: {psnr}")

        # Update the canvas size to fit the filtered frame
        self.update_canvas_size(filtered_frame)
        return filtered_frame


class Filter4Frame(VideoFrame.VideoFrame):

    def apply_filter(self, frame, d_frame, d_output, stream):
        """
        Filter: Bilateral Filter
        """
        frame = frame.astype(np.float32)
        d_frame = cuda.to_device(frame)
        
        # Define the block and grid dimensions
        TILE_SIZE = 16
        threads_per_block = (TILE_SIZE, TILE_SIZE)
        blocks_per_grid_x = math.ceil((frame.shape[0] + TILE_SIZE - 1)/ threads_per_block[0])
        blocks_per_grid_y = math.ceil((frame.shape[1] + TILE_SIZE - 1) / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Define sigma values
        sigma_s = 10.0
        sigma_r = 0.5
                
        # Create CUDA events for timing
        start_event = cuda.event()
        end_event = cuda.event()

        # Record the start event
        start_event.record()
        
        # Launch the kernel
        bilateral_filter[blocks_per_grid, threads_per_block, stream](d_frame, d_output, sigma_s, sigma_r)
                
        cuda.synchronize()  # wait for all threads to complete. The copy to the host performs an implicit synchronization, so the call to cuda.syncronize is not really necessary.
        
        # Record the end event
        end_event.record()

        # Wait for the end event to complete
        end_event.synchronize()

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = cuda.event_elapsed_time(start_event, end_event)

        # Copy the final result back to the host
        result = d_output.copy_to_host()
        
        # Convert the result to uint8
        filtered_frame = np.clip(result, 0, 255).astype(np.uint8)   # Clip may be not necessary

        # Log the results
        filter4_logger.info(f"Timestamp: {datetime.now()}, EXECUTION TIME ms: {elapsed_time_ms}")

        return filtered_frame
    
    
camera_frame = OriginalFrame()
filter_1_frame = Filter1Frame(None)
filter_2_frame = Filter2Frame(None)
filter_3_frame = Filter3Frame(None)
filter_4_frame = Filter4Frame(None)
        
def main():


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