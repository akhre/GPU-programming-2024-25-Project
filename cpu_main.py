from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image as Pil_image, ImageTk as Pil_imageTk, ImageFilter
import cv2
import numpy as np
import cpu.VideoFrame as VideoFrame
from skimage.filters import laplace
import math
import logging
import time
import cpu.BicubicInterpolation as BicubicInterpolation
import cpu.BilateralFiltering as BilateralFiltering
import cpu.GaussianFilter as GaussianFilter
import os # Used to get the video name
from datetime import datetime # Used in the logs to timestamp the execution

RESULTS_DIR = "results/cpu" # Directory to store the results
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
    # Ensure filtered_frame is in 8-bit unsigned integer format and has 3 channels
    filtered_frame_8u = cv2.convertScaleAbs(filtered_frame)
    if len(filtered_frame_8u.shape) == 2 or filtered_frame_8u.shape[2] == 1:
        filtered_frame_8u = cv2.cvtColor(filtered_frame_8u, cv2.COLOR_GRAY2BGR)
    
    filtered_gray = cv2.cvtColor(filtered_frame_8u, cv2.COLOR_BGR2GRAY)

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


    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class Filter1Frame(VideoFrame.VideoFrame):
    def apply_filter(self, frame):
        """
        Filter: Laplacian - CPU
        """
        start_time = time.time()
        # Convert the frame to grayscale
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered_frame = laplace(frame)
        filtered_frame = cv2.convertScaleAbs(filtered_frame)
        # self.photo = Pil_imageTk.PhotoImage(image=Pil_image.fromarray(filtered_frame))
        # self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        
        stop_time = time.time()
        elapsed_time_ms = (stop_time - start_time) * 1000
    
        # Log the results
        filter1_logger.info(f"Timestamp: {datetime.now()}, EXECUTION TIME ms: {elapsed_time_ms}")

        return filtered_frame


class Filter2Frame(VideoFrame.VideoFrame):
    def apply_filter(self, frame):
        """
        Filter: Gaussian Blur - CPU
        """
        start_time = time.time()
        filtered_frame = GaussianFilter.GaussianBlurImage(frame, 2.0)
        stop_time = time.time()
        mse, psnr = measure_distortion(frame, filtered_frame)
        elapsed_time_ms = (stop_time - start_time) * 1000
        filter2_logger.info(f"Timestamp: {datetime.now()}, EXECUTION TIME ms: {elapsed_time_ms}")
        return filtered_frame


class Filter3Frame(VideoFrame.VideoFrame):
    def apply_filter(self, frame):
        """
        Filter: Bicubic interpolation - CPU
        """
        # Scale factor for the bicubic interpolation
        scale = 2.0
        # Coefficient
        a = -1/2
        start_time = time.time()
        filtered_frame = BicubicInterpolation.bicubic(frame, scale, a)
        stop_time = time.time()
        elapsed_time_ms = (stop_time - start_time) * 1000
        # Convert frame to 8-bit unsigned integer format
        frame_8u = cv2.convertScaleAbs(filtered_frame)
        mse, psnr = measure_distortion(frame, filtered_frame)
        # Update the canvas size to fit the filtered frame
        self.update_canvas_size(filtered_frame)
        
        filter3_logger.info(f"Timestamp: {datetime.now()}, EXECUTION TIME ms: {elapsed_time_ms}, MSE: {mse}, PSNR: {psnr}")

        return frame_8u


class Filter4Frame(VideoFrame.VideoFrame):

    def apply_filter(self, frame):
        '''
        Filter: Bilateral filtering - CPU
        '''
        sigma_s = 10.0
        sigma_r = 0.5
        start_time = time.time()
        filtered_frame = BilateralFiltering.bilateral_filter(frame, sigma_s, sigma_r)
        # Convert the filtered frame to uint8
        filtered_frame = cv2.convertScaleAbs(filtered_frame)
        stop_time = time.time()
        elapsed_time_ms = (stop_time - start_time) * 1000

        # Measure distortion
        mse, psnr = measure_distortion(frame, filtered_frame)

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
root.title("CPU Video Filters")
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

# camera_frame.screens.append(filter_1_frame)
# camera_frame.screens.append(filter_2_frame)
# camera_frame.screens.append(filter_3_frame)
camera_frame.screens.append(filter_4_frame)
# camera_frame.update()


root.mainloop()
