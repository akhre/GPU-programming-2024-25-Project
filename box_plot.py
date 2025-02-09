# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Read the dataset from the files in the dataset folder:
DATASET_FOLDER = 'dataset/'
FILTERS = ['filter_bicubic_interpolation', 'filter_bilateral', 'filter_laplacian', 'filter_gaussian_blur']


data = {}


def plot(filter):

    
    input_folder = os.path.join(DATASET_FOLDER + filter + '/')
    
    for file_name in os.listdir(input_folder):
        if not file_name.endswith('.data'):
            continue
        # Read the content of the log file
        df = pd.read_csv(os.path.join(input_folder, file_name), sep=';', header=0)
        elapsed_time = df.iloc[1:, 1]  # Skip the header and get the second column
        data[file_name.replace('.data', '')] = elapsed_time # Store the elapsed time in the data dictionary where the key is the implementation used
        
    # Create a box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data.values(), patch_artist=True)
    plt.xticks(range(1, len(data) + 1), data.keys(), rotation=45)
    plt.xlabel('File Name')
    plt.ylabel('Elapsed Time')
    plt.title(f'Box Plot for {filter}')
    
    # show plot
    plt.show()
    
plot('filter_laplacian')