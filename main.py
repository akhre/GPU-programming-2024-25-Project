from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image as Pil_image, ImageTk as Pil_imageTk
import cv2
import numpy as np
from numba import cuda
import VideoFrame
from Filters.BicubicInterpolation import bicubic_interpolation
from Filters.BilateralFiltering import bilateral_filter
from Filters.GaussianBlur import compute_gaussian_kernel
import math
import logging

# Configure logging for each filter
logging.basicConfig(level=logging.INFO)
filter1_logger = logging.getLogger("Filter1")
filter2_logger = logging.getLogger("Filter2")
filter3_logger = logging.getLogger("Filter3")
filter4_logger = logging.getLogger("Filter4")


filter1_handler = logging.FileHandler("filter1.log", "w")
filter2_handler = logging.FileHandler("filter2.log", "w")
filter3_handler = logging.FileHandler("filter3.log", "w")
filter4_handler = logging.FileHandler("filter4.log", "w")

filter1_logger.addHandler(filter1_handler)
filter2_logger.addHandler(filter2_handler)
filter3_logger.addHandler(filter3_handler)
filter4_logger.addHandler(filter4_handler)


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
        max_kernel_size = 21    # Define the maximum possible size for the Gaussian kernel in order to allocate the shared memory.
        sigma = 2.0
        kernel_size = 2 * int(4 * sigma + 0.5) + 1
        if kernel_size > max_kernel_size:
            raise ValueError("Kernel size is too large")
        
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


@cuda.jit
def process_frame_gpu_filtro_3(frame, output):
    x, y = cuda.grid(2)  # Returns the 2D grid indices (x, y) of the current thread
    if (x < frame.shape[0] and y < frame.shape[1]):  # Check if the thread is within the bounds of the frame
        output[x, y] = bicubic_interpolation(frame[x, y])


def process_frame_gpu_filtro_4(frame, sigma_s, sigma_r):
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
    # Launch the kernel
    bilateral_filter[blocks_per_grid, threads_per_block](frame, result_device, sigma_s, sigma_r)

    # Copy the final result back to the host
    result = result_device.copy_to_host()
    
    # Convert the result to uint8
    result = result.astype(np.uint8)   # Clip may be not necessary
    return result

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
    def __init__(self, parent):
        super().__init__(parent)  # Call the parent class constructor
        self.screens = []
        self.vid = None

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
            for screen in self.screens:
                screen.update(frame)

        self.parent.after(10, self.update)

    def load_video(self, video):
        # print(video)
        self.vid = cv2.VideoCapture(video)  # Open the video file
        self.update()

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class Filter1Frame(VideoFrame.VideoFrame):
    def apply_filter(self, frame, d_frame):
        # Allocate memory on the GPU
        # d_frame = cuda.to_device(np.ascontiguousarray(frame))
        # d_frame = cuda.to_device(frame)
        d_output = cuda.device_array_like(frame)

        # Define the grid and block dimensions
        threads_per_block = (16, 16)

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
        process_frame_gpu_filtro_1[blocks_per_grid, threads_per_block](
            d_frame, d_output
        )

        # Record the end event
        end_event.record()

        # Wait for the end event to complete
        end_event.synchronize()

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = cuda.event_elapsed_time(start_event, end_event)

        # Copy the result back to the host
        filtered_frame = d_output.copy_to_host()

        # Measure distortion
        mse, psnr = measure_distortion(frame, filtered_frame)
        print(f"Filter 1 - MSE: {mse}, PSNR: {psnr}, Execution Time: {elapsed_time_ms} ms")

        # Log the results
        filter1_logger.info(f"MSE: {mse}, PSNR: {psnr}, EXECUTION TIME ms: {elapsed_time_ms}")

        return filtered_frame


class Filter2Frame(VideoFrame.VideoFrame):
    def apply_filter(self, frame, d_frame):
        
        # Allocate memory on the GPU
        d_output = cuda.device_array_like(frame)

        # Define the grid and block dimensions
        threads_per_block = (16, 16)

        blocks_per_grid_x = int(np.ceil((frame.shape[0] + threads_per_block[0] - 1) / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil((frame.shape[1] + threads_per_block[1] - 1) / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        process_frame_gpu_filtro_2[blocks_per_grid, threads_per_block](d_frame, d_output)
        filtered_frame = d_output.copy_to_host()
        mse, psnr = measure_distortion(frame, filtered_frame)
        print(f"Filter 2 - MSE: {mse}, PSNR: {psnr}")
        filter2_logger.info(f"MSE: {mse}, PSNR: {psnr}")
        return filtered_frame


class Filter3Frame(VideoFrame.VideoFrame):

    def apply_filter(self, frame, d_frame):
        # Scale factor for the bicubic interpolation
        scale = 2.0
        # scale = 0.5

        # Allocate memory on the GPU
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
        filter3_logger.info(f"MSE: {mse}, PSNR: {psnr}")
        return filtered_frame


class Filter4Frame(VideoFrame.VideoFrame):

    def apply_filter(self, frame, d_frame):
        # Apply the bilateral filter
        filtered_frame = process_frame_gpu_filtro_4(frame, 15, 30)  # sigma_s = 15.0, sigma_r = 0.1

        # Measure distortion
        mse, psnr = measure_distortion(frame, filtered_frame)
        print(f"Filter 4 - MSE: {mse}, PSNR: {psnr}")

        # Log the results
        filter4_logger.info(f"MSE: {mse}, PSNR: {psnr}")

        return filtered_frame

def button1_click(frame):
    video_source = askopenfilename()
    if video_source:
        frame.load_video(video_source)


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
right_side = Frame(root, bg="#F8F8F8")
right_side.pack(side=RIGHT, padx=20, pady=20)
right_side_left = Frame(right_side, bg="#F8F8F8")
right_side_left.pack(side=LEFT, padx=20, pady=20)
right_side_right = Frame(right_side, bg="#F8F8F8")
right_side_right.pack(side=RIGHT, padx=20, pady=20)


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
filter_1_frame = Filter1Frame(right_side_left)
filter_2_frame = Filter2Frame(right_side_left)
filter_3_frame = Filter3Frame(right_side_right)
filter_4_frame = Filter4Frame(right_side_right)

camera_frame.screens.append(filter_1_frame)
camera_frame.screens.append(filter_2_frame)
camera_frame.screens.append(filter_3_frame)
camera_frame.screens.append(filter_4_frame)
# camera_frame.update()


root.mainloop()
