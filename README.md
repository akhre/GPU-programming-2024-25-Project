# GPU-programming-2024-25-Project
Repository for the GPU Programming course project at the Politecnico di Torino a.y. 2024-2025
Diego Porto
s313169

main.py is used to run the program with the GUI.
main_profiler.py does not have a GUI and it is used only with the profiler with the following command:
```
sudo /usr/local/cuda/bin/nvprof -f --openacc-profiling off --export-profile /home/diego/GPU-programming-2024-25-Project/profile.nvvp /home/diego/GPU-programming-2024-25-Project/main_profiler.py /home/diego/GPU-programming-2024-25-Project/videos/KoNViD_1k_videos/2999049224.mp4
```

The profiler needs java 8 and must be run as:
```
nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
```
