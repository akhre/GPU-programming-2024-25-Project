# GPU-programming-2024-25-Project
Repository for the GPU Programming course project at the Politecnico di Torino a.y. 2024-2025
Diego Porto
s313169

main.py is used to run the program with the GUI.
main_profiler.py does not have a GUI and it is used only with the profiler with the following command:

**Naive:**
```
sudo /usr/local/cuda/bin/nvprof -f --openacc-profiling off --export-profile /home/diego/GPU-programming-2024-25-Project/profile_naive.nvvp /home/diego/GPU-programming-2024-25-Project/main_naive_profiler.py /home/diego/GPU-programming-2024-25-Project/videos/KoNViD_1k_videos/3860380907.mp4
```

**Optimized**:
```
sudo /usr/local/cuda/bin/nvprof -f --openacc-profiling off --export-profile /home/diego/GPU-programming-2024-25-Project/profile_optimized.nvvp /home/diego/GPU-programming-2024-25-Project/main_profiler_optimized.py /home/diego/GPU-programming-2024-25-Project/videos/KoNViD_1k_videos/3860380907.mp4
```


```

The profiler needs java 8 and must be run as:
```
nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
```


To run the code with all the video, the _profiler.py version was used. 
Once the program is terminated the logs are stored in the "results" folder divided by the type of profiler used.
Now, running "create_dataset.py" is possible to create the file .data to use inside "box_plot.py".
