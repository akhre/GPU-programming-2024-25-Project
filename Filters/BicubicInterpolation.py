from numba import cuda

@cuda.jit
def bicubic_interpolation(frame, output, scale):
    x, y = cuda.grid(2) # it uses the formula cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x for the x-dimension and similarly for the y-dimension. The result is a tuple containing two integers, representing the x and y coordinates of the thread.
    if x < output.shape[0] and y < output.shape[1]:
        
        src_x = x / scale
        src_y = y / scale

        x1 = int(src_x)
        y1 = int(src_y)

        dx = src_x - x1
        dy = src_y - y1

        def cubic_weight(t):
            a = -0.5
            t = abs(t)
            if t <= 1:
                return (a + 2) * t**3 - (a + 3) * t**2 + 1
            elif t < 2:
                return a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a
            return 0

        def get_pixel_value(i, j, c):
            if i < 0:
                i = 0
            elif i >= frame.shape[0]:
                i = frame.shape[0] - 1
            if j < 0:
                j = 0
            elif j >= frame.shape[1]:
                j = frame.shape[1] - 1
            return frame[i, j, c]

        for c in range(3):
            value = 0.0
            for m in range(-1, 3):
                for n in range(-1, 3):
                    weight = cubic_weight(m - dx) * cubic_weight(n - dy)
                    value += get_pixel_value(x1 + m, y1 + n, c) * weight
            output[x, y, c] = min(max(int(value), 0), 255)


