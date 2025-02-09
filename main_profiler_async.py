#!/usr/bin/python3

import sys
import cv2
import numpy as np
from numba import cuda
import VideoFrameAsync as VideoFrame
from Filters.BicubicInterpolation import bicubic_interpolation
from Filters.BilateralFiltering import bilateral_filter
from Filters.GaussianBlur import gaussian_filter
from Filters.LaplacianFilter import laplacianFilter
import math
import logging
import os # Used to get the video name
from datetime import datetime # Used in the logs to timestamp the execution

RESULTS_DIR = "results/gpu" # Directory to store the results
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)    # Create the results directory if it does not exist

# VIDEO_PATH = "/media/diego/Volume/Polito/Magistrale/GPU programming/Lab/Project/videos/videoplayback.mp4"
# VIDEO_PATH = "/media/diego/Volume/Polito/Magistrale/GPU programming/Lab/Project/videos/KoNViD_1k_videos/12099271596.mp4"
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
        self.update()

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class Filter1Frame(VideoFrame.VideoFrame):
    async def apply_filter(self, frame, d_frame):
        """
        Filter: Laplacian Filter
        """
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Allocate the new frame on the GPU memory:
        d_frame_gray = cuda.to_device(np.ascontiguousarray(gray_frame))
        # d_frame = cuda.to_device(frame)
        d_output = cuda.device_array_like(frame)
        
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
        laplacianFilter[blocks_per_grid, threads_per_block](
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
    async def apply_filter(self, frame, d_frame):
        """
        Filter: Gaussian Blur
        """
        # Allocate memory on the GPU
        d_output = cuda.device_array_like(frame)

        # Define the grid and block dimensions
        threads_per_block = (16, 16)

        blocks_per_grid_x = int(np.ceil((frame.shape[0] + threads_per_block[0] - 1) / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil((frame.shape[1] + threads_per_block[1] - 1) / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        # Create CUDA events for timing
        start_event = cuda.event()
        end_event = cuda.event()

        # Record the start event
        start_event.record()

        # Launch the kernel
        gaussian_filter[blocks_per_grid, threads_per_block](d_frame, d_output)
        cuda.synchronize()  # wait for all threads to complete. The copy to the host performs an implicit synchronization, so the call to cuda.syncronize is not really necessary.
        
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

    async def apply_filter(self, frame, d_frame):
        # Scale factor for the bicubic interpolation
        scale = 2.0
        # scale = 0.5

        # Allocate memory on the GPU
        # Calculate the dimensions of the scaled frame
        scaled_height = int(frame.shape[0] * scale)
        scaled_width = int(frame.shape[1] * scale)
        # d_output = cuda.device_array_like(frame)
        # Allocate memory for the scaled frame
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
        bicubic_interpolation[blocks_per_grid, threads_per_block](d_frame, d_output, scale)
        
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

    async def apply_filter(self, frame, d_frame):
        # Bilateral filtering
        result_device = cuda.device_array((frame.shape[0], frame.shape[1], 3), dtype=np.float32)

        # Define the block and grid dimensions
        threads_per_block = (16, 16)
        blocks_per_grid_x = math.ceil(frame.shape[0] / threads_per_block[0])
        blocks_per_grid_y = math.ceil(frame.shape[1] / threads_per_block[1])
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
        bilateral_filter[blocks_per_grid, threads_per_block](frame, result_device, sigma_s, sigma_r)
                
        cuda.synchronize()  # wait for all threads to complete. The copy to the host performs an implicit synchronization, so the call to cuda.syncronize is not really necessary.
        
        # Record the end event
        end_event.record()

        # Wait for the end event to complete
        end_event.synchronize()

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = cuda.event_elapsed_time(start_event, end_event)

        # Copy the final result back to the host
        result = result_device.copy_to_host()
        
        # Convert the result to uint8
        filtered_frame = result.astype(np.uint8)   # Clip may be not necessary

        # Log the results
        filter4_logger.info(f"Timestamp: {datetime.now()}, EXECUTION TIME ms: {elapsed_time_ms}")

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