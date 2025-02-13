import glob

def get_average_frame_time(file_list):
    total_time = 0
    total_frame = 0
    for file_name in file_list:
        with open(file_name, 'r') as file:
            # line = file.readline()
            while True:
                line = file.readline()
                if not line:
                    break
                fields = line.split(",")
                field_ex_time = fields[1].split(":")
                ex_time = field_ex_time[1].strip()
                total_time += float(ex_time)
                total_frame += 1
    avg_time = total_time / total_frame
    return total_frame, avg_time

folders = ['results/cpu/','results/gpu_naive/', 'results/gpu_optimized/']
# folders = ['results/gpu_naive/', 'results/gpu_optimized/']
for folder in folders:
    bicubic_file_list  = glob.glob(folder + '/*_filter_bicubic_interpolation.log')
    total_frame, avg_time = get_average_frame_time(bicubic_file_list)

    w = open(folder + '/filter_bicubic_interpolation.data','w')
    w.write(f'{total_frame};{avg_time}\n')
    w.close()

    bilateral_file_list  = glob.glob(folder + '/*_filter_bilateral.log')
    total_frame, avg_time = get_average_frame_time(bilateral_file_list)

    w = open(folder + '/filter_bilateral.data','w')
    w.write(f'{total_frame};{avg_time}\n')
    w.close()

    gaussian_blur_file_list  = glob.glob(folder + '/*_filter_gaussian_blur.log')
    total_frame, avg_time = get_average_frame_time(gaussian_blur_file_list)

    w = open(folder + '/filter_gaussian_blur.data','w')
    w.write(f'{total_frame};{avg_time}\n')
    w.close()

    laplacian_file_list  = glob.glob(folder + '/*_filter_laplacian.log')
    total_frame, avg_time = get_average_frame_time(laplacian_file_list)

    w = open(folder + '/filter_laplacian.data','w')
    w.write(f'{total_frame};{avg_time}\n')
    w.close()