import os
from datetime import datetime
import numpy as np
import winsound

total_re_do_times = 10
dataset_name = 'paviaU'
re_do_times = 1
txt_filename = dataset_name + "_" + "{0:%Y_%m_%d_%H_%M_%S.txt}".format(datetime.now())
output_dir = dataset_name + "_" + "{0:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
per_class = 30

if dataset_name == 'paviaU':
    spatial_para = 0.001
    spectral_para = 0.01

    loss_para_1 = 0.045
    loss_para_2 = 0.045

    combine_para = 0.5
elif dataset_name == 'IP':
    spatial_para = 0.1
    spectral_para = 0.1

    loss_para_1 = 0.9
    loss_para_2 = 0.9
    combine_para = 0.6
elif dataset_name == 'SA':
    spatial_para = 0.01
    spectral_para = 0.1

    loss_para_1 = 0.1
    loss_para_2 = 0.9
    combine_para = 0.5
else:
    raise NotImplementedError

for i in range(re_do_times, total_re_do_times + 1):
    cmd_line = "python trainGCN.py " + "-r " + str(
        re_do_times) + " -f " + txt_filename + " -d " + output_dir + " -t " + dataset_name + " --spatial_para " + str(spatial_para) + " --spectral_para " + str(
        spectral_para) + " --loss_para_1 " + str(
        loss_para_1) + " --loss_para_2 " + str(
        loss_para_2) + " --combine_para " + str(combine_para) + " --per_class " + str(per_class)
    re_do_times = re_do_times + 1
    os.system(cmd_line)

with open(txt_filename, 'r') as f:
    content = f.read()

results_str = content.split('\n')

if dataset_name == 'paviaU':
    results_npy = np.ones(shape=(total_re_do_times, 13))
elif dataset_name == 'IP':
    results_npy = np.ones(shape=(total_re_do_times, 20))
elif dataset_name == 'SA':
    results_npy = np.ones(shape=(total_re_do_times, 20))
else:
    raise NotImplementedError

for i, element in enumerate(results_str):
    if i < len(results_str) - 1:
        results = element.split(' ')
        results_npy[i, :] = np.array(results, dtype=np.float32)

avg_results = np.mean(results_npy, axis=0)
std_results = np.std(results_npy, axis=0)

final_result = ''
for i, element in enumerate(avg_results):
    final_result = final_result + '%.4f' % element + '±' + '%.4f' % std_results[i] + ' '

print(final_result)

with open(txt_filename, "a") as file:
    file.write('\n\n' + final_result + '\n\n' + cmd_line)

winsound.Beep(600, 2000)
