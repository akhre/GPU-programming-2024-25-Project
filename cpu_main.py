from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image as Pil_image, ImageTk as Pil_imageTk, ImageFilter
import cv2
import numpy as np
import cpu.VideoFrame as VideoFrame 
import math
import logging
import time
import cpu.BicubicInterpolation as BicubicInterpolation


# Configure logging for each filter
logging.basicConfig(level=logging.INFO)
filter1_logger = logging.getLogger("Filter1")
filter2_logger = logging.getLogger("Filter2")
filter3_logger = logging.getLogger("Filter3")
filter4_logger = logging.getLogger("Filter4")


filter1_handler = logging.FileHandler("cpu_filter1.log", "w")
filter2_handler = logging.FileHandler("cpu_filter2.log", "w")
filter3_handler = logging.FileHandler("cpu_filter3.log", "w")
filter4_handler = logging.FileHandler("cpu_filter4.log", "w")

filter1_logger.addHandler(filter1_handler)
filter2_logger.addHandler(filter2_handler)
filter3_logger.addHandler(filter3_handler)
filter4_logger.addHandler(filter4_handler)



def measure_distortion(original_frame, filtered_frame):
    # Convert frames to grayscale
    original_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    filtered_frame_8u = cv2.convertScaleAbs(filtered_frame)
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

        filtered_frame = frame
        # Measure distortion
        mse, psnr = measure_distortion(frame, filtered_frame)
        # print(f"Filter 1 - MSE: {mse}, PSNR: {psnr}, Execution Time: {elapsed_time_ms} ms")

        # Log the results
        # filter1_logger.info(f"MSE: {mse}, PSNR: {psnr}, EXECUTION TIME ms: {elapsed_time_ms}")

        return filtered_frame


class Filter2Frame(VideoFrame.VideoFrame):
    def apply_filter(self, frame):
        """
        Gaussian filter implemented via CPU
        """
        start_time = time.time()
        filtered_frame = cv2.GaussianBlur(frame, (21, 21), 0)
        stop_time = time.time()
        mse, psnr = measure_distortion(frame, filtered_frame)
        elapsed_time_ms = (stop_time - start_time)
        print(f"Filter 2 - MSE: {mse}, PSNR: {psnr}, Elapsed Time: {elapsed_time_ms} ms")
        filter2_logger.info(f"MSE: {mse}, PSNR: {psnr}, Elapsed Time: {elapsed_time_ms} ms")
        return filtered_frame


class Filter3Frame(VideoFrame.VideoFrame):
    def apply_filter(self, frame):
        """
        Bicubic interpolation implemented via CPU
        """
        # Scale factor for the bicubic interpolation
        scale = 2.0
        # Coefficient
        a = -1/2
        start_time = time.time()
        filtered_frame = BicubicInterpolation.bicubic(frame, scale, a)
        stop_time = time.time()
        elapsed_time_ms = (stop_time - start_time)
        # Convert frame to 8-bit unsigned integer format
        frame_8u = cv2.convertScaleAbs(filtered_frame)
        # mse, psnr = measure_distortion(frame, filtered_frame)
        print(f"Filter 3 - MSE: - , PSNR: - , Elapsed Time: {elapsed_time_ms} ms")
        filter3_logger.info(f"Filter 3 - MSE: - , PSNR: - , Elapsed Time: {elapsed_time_ms} ms")
        return frame_8u


class Filter4Frame(VideoFrame.VideoFrame):

    def apply_filter(self, frame):
        # Apply the bilateral filter
        filtered_frame = frame

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
# filter_1_frame = Filter1Frame(right_side_left)
filter_2_frame = Filter2Frame(right_side_left)
filter_3_frame = Filter3Frame(right_side_right)
# filter_4_frame = Filter4Frame(right_side_right)

# camera_frame.screens.append(filter_1_frame)
camera_frame.screens.append(filter_2_frame)
camera_frame.screens.append(filter_3_frame)
# camera_frame.screens.append(filter_4_frame)
# camera_frame.update()


root.mainloop()
