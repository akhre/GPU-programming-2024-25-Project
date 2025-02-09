#!/bin/bash

# Directory containing the videos
VIDEO_DIR="./videos/KoNViD_1k_videos"

# Loop through all video files in the directory
for video in "$VIDEO_DIR"/*; do
    # Call the main_profiler.py program with the video file as an argument
    /usr/bin/python3 ./main_profiler.py "$video"
done