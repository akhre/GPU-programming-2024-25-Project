from tkinter import *
from PIL import Image as Pil_image, ImageTk as Pil_imageTk
import cv2
from numba import cuda
import numpy as np


class VideoFrame:
    def __init__(self, parent):
        self.parent = parent
        self.canvas = Canvas(self.parent, width=800, height=600)  # Set the canvas size
        self.canvas.pack()
        
    def update_canvas_size(self, frame):
        self.canvas.config(width=frame.shape[1], height=frame.shape[0])
        
    
    def update(self, frame):
        d_frame = cuda.to_device(np.ascontiguousarray(frame))
        # Apply selected filter to the frame
        frame = self.apply_filter(frame, d_frame)
        self.photo = Pil_imageTk.PhotoImage(image=Pil_image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        # Position the image in the center with padding
        x = (self.canvas.winfo_width() - self.photo.width()) / 2
        y = (self.canvas.winfo_height() - self.photo.height()) / 2
        self.canvas.create_image(x, y, image=self.photo, anchor=NW)  
