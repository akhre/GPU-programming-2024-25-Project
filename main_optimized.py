from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image as Pil_image, ImageTk as Pil_imageTk
import cv2
import numpy as np
from numba import cuda
import VideoFrameOptimized as VideoFrame
from Filters.BicubicInterpolation import bicubic_interpolation
from Filters.BilateralFiltering_optimized import bilateral_filter
from Filters.GaussianBlur import gaussian_filter
from Filters.LaplacianFilter import laplacianFilter
import math
import logging
import os # Used to get the video name
from datetime import datetime # Used in the logs to timestamp the execution

RESULTS_DIR = "results/gpu_optimized" # Directory to store the results
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)    # Create the results directory if it does not exist


# Configure logging for each filter
logging.basicConfig(level=logging.INFO)
filter1_logger = logging.getLogger("Filter1")
filter2_logger = logging.getLogger("Filter2")
filter3_logger = logging.getLogger("Filter3")
filter4_logger = logging.getLogger("Filter4")

def get_video_name(video_path):
    return os.path.splitext(os.path.basename(video_path))[0]


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
    def __init__(self, parent):
        super().__init__(parent)  # Call the parent class constructor
        self.screens = []
        self.vid = None
        
        self.d_output_filter_1 = None
        self.d_output_filter_2 = None
        self.d_output_filter_3 = None
        self.d_output_filter_4 = None

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = Pil_imageTk.PhotoImage(
                image=Pil_image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            )
            # Position the image in the center with padding
            x = (self.canvas.winfo_width() - self.photo.width()) / 2
            y = (self.canvas.winfo_height() - self.photo.height()) / 2
            self.canvas.create_image(x, y, image=self.photo, anchor=NW)
            self.apply_filters(frame)
            
            
            
            # for screen in self.screens:
            #     screen.update(frame)
        else:
            self.reset_frame()  # Reset the frame when the video ends
            self.vid.release()
            return

        self.parent.after(10, self.update)

    def load_video(self, video):
        self.vid = cv2.VideoCapture(video)  # Open the video file
        self.update()
    
    def reset_frame(self):
        # Set a blank frame or a specific image to indicate the end of the video
        blank_frame = np.zeros((self.canvas.winfo_height(), self.canvas.winfo_width(), 3), dtype=np.uint8)
        self.photo = Pil_imageTk.PhotoImage(image=Pil_image.fromarray(blank_frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
      
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
            
        # d_output = cuda.device_array_like(frame, stream_copy)
        # d_output_filter_3 = 
        # d_output_filter_4 = cuda.device_array((frame.shape[0], frame.shape[1], 3), dtype=np.float32, stream=stream_copy)
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
        
        
        # NO NO Asynchronously copy the results back to the host
        # NO LOL filtered_frame2 = d_output_filter_2.copy_to_host(stream=stream_filter_2)
        
        
        
        
        # filter_2_frame.apply_filter(frame, d_frame, stream_filter_2)
        
        # return filtered_frame2
        
        

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
        # Allocate memory on the GPU
        # d_output = cuda.device_array_like(frame)

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
        gaussian_filter[blocks_per_grid, threads_per_block, stream](d_frame, d_output)
        # cuda.synchronize()  # wait for all threads to complete. The copy to the host performs an implicit synchronization, so the call to cuda.syncronize is not really necessary.
        
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
        # Bilateral filtering
        # d_output = cuda.device_array((frame.shape[0], frame.shape[1], 3), dtype=np.float32)

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
        bilateral_filter[blocks_per_grid, threads_per_block, stream](frame, d_output, sigma_s, sigma_r)
                
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
        filtered_frame = result.astype(np.uint8)   # Clip may be not necessary

        # Log the results
        filter4_logger.info(f"Timestamp: {datetime.now()}, EXECUTION TIME ms: {elapsed_time_ms}")

        return filtered_frame

def button1_click(frame):
    video_source = askopenfilename(initialdir="./videos/KoNViD_1k_videos", title="Select a video file")
    if video_source:
        video_name = get_video_name(video_source)
        frame.load_video(video_source)
        logging.basicConfig(level=logging.INFO)

        filter1_handler = logging.FileHandler(f"{RESULTS_DIR}/{video_name}_filter_laplacian.log", "w")
        filter2_handler = logging.FileHandler(f"{RESULTS_DIR}/{video_name}_filter_gaussian_blur.log", "w")
        filter3_handler = logging.FileHandler(f"{RESULTS_DIR}/{video_name}_filter_bicubic_interpolation.log", "w")
        filter4_handler = logging.FileHandler(f"{RESULTS_DIR}/{video_name}_filter_bilateral.log", "w")

        filter1_logger.addHandler(filter1_handler)
        filter2_logger.addHandler(filter2_handler)
        filter3_logger.addHandler(filter3_handler)
        filter4_logger.addHandler(filter4_handler)



# Create the main window
root = Tk()
root.title("GPU Video Filters")
# root.geometry('1200x800')
root.geometry("3840x2160")  # Set the window size to 4K
root.configure(background="#F8F8F8")


text_label = Label(root, text="Video Filters", fg="black", bg="#F8F8F8")
text_label.pack(pady=(10, 10))
text_label.config(font=("Vandana", 25))

left_side = Frame(root, bg="#F8F8F8")
left_side.pack(side=LEFT, padx=20, pady=20)
left_side_top = Frame(left_side, bg="#F8F8F8")
left_side_top.pack(side=TOP, padx=20, pady=20)
left_side_bottom = Frame(left_side, bg="#F8F8F8")
left_side_bottom.pack(side=BOTTOM, padx=20, pady=20)
Label(left_side_bottom, text="Original Video", bg="#F8F8F8").pack()

right_side = Frame(root, bg="#F8F8F8")
right_side.pack(side=RIGHT, padx=20, pady=20)

right_top_left_frame = Frame(right_side, bg="#F8F8F8")
right_top_left_frame.grid(row=0, column=0, padx=10, pady=10)
Label(right_top_left_frame, text="Laplacian Filter", bg="#F8F8F8").pack()

right_top_right_frame = Frame(right_side, bg="#F8F8F8")
right_top_right_frame.grid(row=0, column=1, padx=10, pady=10)
Label(right_top_right_frame, text="Gaussian Blur", bg="#F8F8F8").pack()

right_bottom_left_frame = Frame(right_side, bg="#F8F8F8")
right_bottom_left_frame.grid(row=1, column=0, padx=10, pady=10)
Label(right_bottom_left_frame, text="Bicubic Interpolation", bg="#F8F8F8").pack()

right_bottom_right_frame = Frame(right_side, bg="#F8F8F8")
right_bottom_right_frame.grid(row=1, column=1, padx=10, pady=10)
Label(right_bottom_right_frame, text="Bilateral Filter", bg="#F8F8F8").pack()


# Add button for open file dialog
button1 = Button(
    left_side_top,
    text="Load video",
    command=lambda: button1_click(camera_frame),
    bg="red",
    width=15,
    height=3,
)
button1.pack(side=LEFT, padx=10, pady=10)

# Add live camera feed to the left side of the GUI
camera_frame = OriginalFrame(left_side_bottom)

filter_1_frame = Filter1Frame(right_top_left_frame)
filter_2_frame = Filter2Frame(right_top_right_frame)
filter_3_frame = Filter3Frame(right_bottom_left_frame)
filter_4_frame = Filter4Frame(right_bottom_right_frame)

camera_frame.screens.append(filter_1_frame)
camera_frame.screens.append(filter_2_frame)
camera_frame.screens.append(filter_3_frame)
camera_frame.screens.append(filter_4_frame)
# camera_frame.update()


root.mainloop()