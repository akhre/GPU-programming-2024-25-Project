import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_value(file):
    with open(file, 'r') as f:
        return f.readline().split(';')[1]

filters = ['filter_bicubic_interpolation', 'filter_bilateral', 'filter_laplacian', 'filter_gaussian_blur']
folders = ['results/cpu/','results/gpu_naive/', 'results/gpu_optimized/']
# folders = ['results/gpu_naive/', 'results/gpu_optimized/']

cpu_times = []
gpu_naive_times = []
gpu_optimized_times = []

for filter in filters:
    cpu_times.append(float(read_value(folders[0] + filter + '.data')))
    gpu_naive_times.append(float(read_value(folders[1] + filter + '.data')))
    gpu_optimized_times.append(float(read_value(folders[2] + filter + '.data')))

filter_label = ["Bicubic", "Bilateral", "Laplacian", "Gaussian blur"]


# Create the data frame: 
df = pd.DataFrame({
    "Filter": np.tile(filter_label, 3),
    "Time (ms)": cpu_times + gpu_naive_times + gpu_optimized_times,
    "Hardware": ["CPU"] * 4 + ["GPU NAIVE"] * 4 + ["GPU OPTIMIZED"] * 4
})

# Set the style
sns.set_style("whitegrid")

# Creazione del grafico a barre raggruppate
plt.figure(figsize=(8, 5))
ax = sns.barplot(x="Filter", y="Time (ms)", hue="Hardware", data=df, palette=["red", "blue", "green"])

# Set a logarithmic scale for the y-axis
ax.set_yscale("log")

# Add labels to the bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.0f}", 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha="center", va="bottom", fontsize=10, color="black")


# plt.title("Confronto Prestazioni CPU vs GPU vs TPU (Scala Logaritmica)", fontsize=14)
plt.title("Comparison between execution time per frame for different hardware", fontsize=14)
plt.xlabel("Filters", fontsize=12)
plt.ylabel("Execution time (ms)", fontsize=12)
plt.legend(title="Hardware")
plt.show()
