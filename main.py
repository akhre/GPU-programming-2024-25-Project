from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image as Pil_image, ImageTk as Pil_imageTk
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
    # threads_per_block = (16, 16)
    blockSize = 16
    threads_per_block = (blockSize, blockSize)
    # Per coprire l'intera immagine con dei threads dobbiamo prendere la dimensione del frame e dividerla per il numero di thread in ciascun blocco.
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
    def apply_filter(self, frame):
        # Allocate memory on the GPU
        d_frame = cuda.to_device(np.ascontiguousarray(frame))
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
        filter2_logger.info(f"MSE: {mse}, PSNR: {psnr}")
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
        filter3_logger.info(f"MSE: {mse}, PSNR: {psnr}")
        return filtered_frame


class Filter4Frame(VideoFrame.VideoFrame):

    def apply_filter(self, frame):
        # Apply the bilateral filter
        filtered_frame = process_frame_gpu_filtro_4(frame, 15, 30)  # sigma_s = 15.0, sigma_r = 0.1
        # filtered_frame = frame
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
