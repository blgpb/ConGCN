import numpy as np
import scipy.io as scio


def Con2Numpy(path, filename, var_name):
    dataFile = path + filename
    data = scio.loadmat(dataFile)
    x = data[var_name]
    x1 = x.astype(float)
    return x1


def load_HSI_data(data_name, re_do_times):
    Data = dict()
    img_gyh = data_name + '_gyh'
    img_gt = data_name + '_gt'
    if data_name == 'IP':
        path = './/data//IndianPines//'
    elif data_name == 'SA':
        path = './/data//SA//'
    elif data_name == 'paviaU':
        path = './/data//paviaU//'
    Data['useful_sp_lab'] = np.array(Con2Numpy(path, 'useful_sp_lab', 'useful_sp_lab'), dtype='int')
    Data[img_gt] = np.array(Con2Numpy(path, img_gt, img_gt), dtype='int')
    Data[img_gyh] = Con2Numpy(path, img_gyh, img_gyh)
    Data['trpos'] = np.array(Con2Numpy(path, 'trpos_' + str(re_do_times) + '_per_class', 'trpos'), dtype='int')

    return Data
