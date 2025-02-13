from numba import cuda, float32, int32

@cuda.jit
def cubic_weight(t):
    """
    Cubic weight function: Calculates how much to "blend" nearby pixels using a smooth curve
    """
    a = -0.5
    t = abs(t)
    if t <= 1:
        return (a + 2) * t**3 - (a + 3) * t**2 + 1
    elif t < 2:
        return a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a
    return 0

@cuda.jit
def bicubic_interpolation(frame, output, scale):
    x, y = cuda.grid(2)
    
    if x >= output.shape[0] or y >= output.shape[1]:
        return
    
    src_x = x / scale
    src_y = y / scale
    x1 = int(src_x)
    y1 = int(src_y)
    dx = src_x - x1
    dy = src_y - y1
    
    # Use CUDA local arrays to store weights
    wx = cuda.local.array(4, float32)
    wy = cuda.local.array(4, float32)
    
    for offset in range(4):
        wx[offset] = cubic_weight((offset - 1) - dx)
        wy[offset] = cubic_weight((offset - 1) - dy)
    
    # Precompute coordinates using arrays
    xs = cuda.local.array(4, int32)
    ys = cuda.local.array(4, int32)
    max_x = frame.shape[0] - 1
    max_y = frame.shape[1] - 1
    
    for i in range(4):
        # Clamp to image boundaries with min/max instead of if conditions
        xs[i] = min(max(x1 + (i - 1), 0), max_x)
        ys[i] = min(max(y1 + (i - 1), 0), max_y)
    
    for c in range(3):
        val = 0.0
        for i in range(4):
            for j in range(4):
                val += frame[xs[i], ys[j], c] * wx[i] * wy[j]
        output[x, y, c] = int(max(0, min(val, 255)))