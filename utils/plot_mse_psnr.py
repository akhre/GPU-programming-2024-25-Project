import os
import re

import matplotlib.pyplot as plt

"""
Program to plot the data for the MSE and PSNR of Bicubic interpolation filter applied to each frame
"""

def read_log_file(filename):
    mse_values = []
    psnr_values = []
    with open(filename, 'r') as file:
        for line in file:
            if "MSE" in line and "PSNR" in line:
                parts = line.strip().split(',')
                mse = float(parts[2].split(':')[1].strip())
                psnr = float(parts[3].split(':')[1].strip())
                mse_values.append(mse)
                psnr_values.append(psnr)
    return mse_values, psnr_values

# Read log files
folders = ['results/cpu/','results/gpu_naive/', 'results/gpu_optimized/']

log_files = []
for folder in folders:
    for file in os.listdir(folder):
        if re.match(r'.*_filter_bicubic_interpolation\.log$', file):
            log_files.append(os.path.join(folder, file))

mse_values = []
psnr_values = []
for log_file in log_files:
    mse, psnr = read_log_file(log_file)
    mse_values.append(mse)
    psnr_values.append(psnr)


# mse1, psnr1 = read_log_file('filter1.log')
# mse2, psnr2 = read_log_file('filter2.log')
# mse3, psnr3 = read_log_file('filter3.log')
# mse4, psnr4 = read_log_file('filter4.log')

# Plot MSE values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# plt.plot(mse1, label='Filter 1')
# plt.plot(mse2, label='Filter 2')
plt.plot(mse_values, label='Bicubic interpolation 3')
# plt.plot(mse4, label='Filter 4')
plt.xlabel('Frame')
plt.ylabel('MSE')
plt.title('Mean Squared Error (MSE)')
plt.legend()

# Plot PSNR values
plt.subplot(1, 2, 2)
# plt.plot(psnr1, label='Filter 1')
# plt.plot(psnr2, label='Filter 2')
plt.plot(psnr_values, label='Filter 3')
# plt.plot(psnr4, label='Filter 4')
plt.xlabel('Frame')
plt.ylabel('PSNR (dB)')
plt.title('Peak Signal-to-Noise Ratio (PSNR)')
plt.legend()

plt.tight_layout()
plt.show()
