"""
Program to extract the data from the log files and parse this information to create a box plot

For each filter is created a folder where to store the results.
In each folder, based on the implementation, it is computed an output file containing the average elapsed time per frame per each video.

Example:
    - filter_laplacian/cpu.data : contains the average elapsed time per frame for the laplacian filter implemented via CPU
    - filter_bilateral/gpu_naive.data : contains the average elapsed time per frame for the bilateral filter implemented via GPU with the naive method.

The content of each file is a CSV separated with ';'. An header is present to identify the columns (video name and average elapsed time).
"""

import os

INPUT_FOLDER = 'results/'
OUTPUT_FOLDER = 'dataset/'
FILTERS = ['filter_bicubic_interpolation', 'filter_bilateral', 'filter_laplacian', 'filter_gaussian_blur']

def compute_average_per_video(type_of_implementation):
    """
    Compute the average elapsed time per frame per video for each filter.
    """
    input_folder = os.path.join(INPUT_FOLDER + type_of_implementation + '/')
    # output_folder = os.path.join(OUTPUT_FOLDER + type_of_implementation + '/')


    
    # Initilize the output file
    for FILTER_NAME in FILTERS:
        # Create the output folder if it does not exist
        if not os.path.exists(os.path.join(OUTPUT_FOLDER + FILTER_NAME)):
            os.makedirs(os.path.join(OUTPUT_FOLDER + FILTER_NAME))
        with open(os.path.join(OUTPUT_FOLDER + FILTER_NAME + '/' + type_of_implementation + '.data'), 'w') as file:
            file.write('Video name; Average elapsed time (ms) per frame\n')
            pass
                
    
    # Iterate over the log files
    for file_name in os.listdir(input_folder):
        if not file_name.endswith('.log'):
            continue
        # Read the content of the log file
        with open(input_folder + file_name, 'r') as file:
            content = file.readlines()
        
        # Extract the video name
        video_name = file_name.split('_')[0]
        
        # Extract the filter number
        filter_name = '_'.join(file_name.split('_')[1:]).split('.')[0]
        
        # Compute the average elapsed time per frame
        elapsed_times = [float(line.split('EXECUTION TIME ms: ')[1].split(',')[0]) for line in content if 'EXECUTION TIME ms: ' in line]
        average_elapsed_time = sum(elapsed_times) / len(elapsed_times)
        
        # Write the average elapsed time per frame to the output file
        with open(os.path.join(OUTPUT_FOLDER + filter_name + '/' + type_of_implementation + '.data'), 'a') as file:
            file.write(video_name +';' + str(average_elapsed_time) + '\n')
            


            
if __name__ == '__main__':
    # Compute the average elapsed time per frame per video for each filter and each implementation
    # compute_average_per_video('cpu')
    compute_average_per_video('gpu_naive')
    # compute_average_per_video('gpu_streams')