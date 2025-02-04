from numba import cuda

info = cuda.get_current_device()
print(info)