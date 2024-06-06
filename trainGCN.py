import os

from funcCNN import *
from CG3Model import GCNModel
from BuildSPInst_A import *

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import scipy.sparse as sp
import time
from train import HGCN_Model
from spatial_augumentation import spatial_augmentation
from spectral_augumentation import drop_feature_weighted_2, drop_feature_weighted_adaptive, \
    spectral_level_graph_augmentation

from datetime import datetime
import argparse

time_start = time.time()

parser = argparse.ArgumentParser(description='Input some parameters.')
parser.add_argument('-r', '--re_do_times', type=int, metavar='', required=True,
                    help='This is the i-th time to retrain the model.')
parser.add_argument('-f', '--txt_filename', type=str, metavar='', required=True, help='This is the file name of txt.')
parser.add_argument('-d', '--output_dir', type=str, metavar='', required=True,
                    help='This is the dir name of output .mat files.')
parser.add_argument('-t', '--dataset_name', type=str, metavar='', required=True, help='This is the name of dataset.')

parser.add_argument('--enable_spatial_augmentation', type=bool, metavar='', default=True,
                    help='whether enable spatial augmentation')
parser.add_argument('--enable_spectral_augmentation', type=bool, metavar='', default=True,
                    help='whether enable spectral augmentation')

parser.add_argument('--spatial_para', type=float, metavar='', required=True,
                    help='the parameter of spatial augmentation')
parser.add_argument('--spectral_para', type=float, metavar='', required=True,
                    help='the parameter of spectral augmentation')

parser.add_argument('--enable_contrastive_loss', type=bool, metavar='', default=True,
                    help='whether to enable contrastive loss')
parser.add_argument('--enable_generative_loss', type=bool, metavar='', default=True,
                    help='whether to enable generative loss')
parser.add_argument('--loss_para_1', type=float, metavar='', required=True,
                    help='the parameter of contrastive loss')
parser.add_argument('--loss_para_2', type=float, metavar='', required=True,
                    help='the parameter of contrastive loss')
parser.add_argument('--combine_para', type=float, metavar='', required=True,
                    help='the parameter of combining GCN and HGCN')
parser.add_argument('--per_class', type=int, metavar='', required=True,
                    help='the number of training examples per class')

args = parser.parse_args()


def sparse_to_tuple(sparse_mx):

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def GCNevaluate(mask1, labels1):
    t_test = time.time()
    outs_val = sess.run([GCNmodel.loss, GCNmodel.accuracy], feed_dict={labels: labels1, mask: mask1})
    return outs_val[0], outs_val[1], (time.time() - t_test)


dataset_name = args.dataset_name

seed = 123
hidden_num = 256
learning_rate = 0.01
epochs = 4000
dropout_all = 0.0001
weight_decay = 0.0
re_do_times = args.re_do_times
txt_filename = args.txt_filename
per_class = args.per_class

enable_spatial_augmentation = args.enable_spatial_augmentation
enable_spectral_augmentation = args.enable_spectral_augmentation


img_gyh = dataset_name + '_gyh'
img_gt = dataset_name + '_gt'
HSI_data = load_HSI_data(dataset_name, per_class)
graph_data = GetInst_A(HSI_data['useful_sp_lab'], HSI_data[img_gyh], HSI_data[img_gt], HSI_data['trpos'])
features_1 = np.array(graph_data.sp_mean, dtype='float32')

adj = graph_data.sp_A
adj = np.array(adj, dtype='float32')
adj = np.squeeze(adj)

features = sp.coo_matrix(features_1)
features = sparse_to_tuple(features)
feature_sp = tf.SparseTensor(features[0], np.array(features[1], dtype='float32'), features[2])

y_train = np.array(graph_data.sp_label, dtype='float32')
y_val = y_train
y_test = y_train
train_mask = np.matlib.reshape(np.array(graph_data.trmask, dtype='bool'),
                               [np.shape(graph_data.trmask)[0], ])
test_mask = np.matlib.reshape(np.array(graph_data.temask, dtype='bool'),
                              [np.shape(graph_data.trmask)[0], ])


num_classes = np.shape(y_train)[1]
num_inst = np.shape(y_train)[0]


input_dim = features[2][1]

support_1 = np.array(graph_data.CalSupport(adj), dtype='float32')
support = sp.coo_matrix(support_1)
support = sparse_to_tuple(support)

num_inst = features[2][0]

trtemask = processmask(train_mask)

placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),
}

dp_fea0 = [placeholders['dropout'], placeholders['num_features_nonzero']]

mask = tf.placeholder("int32", [None])
labels = tf.placeholder("float", [None, num_classes])

paras = dict()
paras['hidden_num'] = hidden_num
paras['weight_decay'] = weight_decay
paras['dataset'] = dataset_name
HGCNModel = HGCN_Model(placeholders, paras, per_class)

y_dim1 = np.argmax(y_train, axis=1)
y_dim = np.ones([num_inst]) * -1
tr_idx = np.argwhere(np.sum(y_train, axis=1) > 0)[:, 0]
y_dim[tr_idx] = y_dim1[tr_idx]

intra_class_idx = []
for i in range(num_classes):
    intra_class_idx.append(np.argwhere(y_dim == i)[:, 0])

train_mat01 = CalCLass01Mat(y_train, train_mask)
mats_intra_inter = CalIntraClassMat01(y_dim1[tr_idx])
num_labeled = int(np.sum(y_train))
mats_intra_inter[0] += np.eye(num_labeled)

np.random.seed(seed)
tf.set_random_seed(seed)
GCNmodel = GCNModel(feature_sp=feature_sp, learning_rate=learning_rate,
                    num_classes=num_classes, support=placeholders['support'],
                    h=hidden_num, input_dim=input_dim,
                    HGCN=HGCNModel, train_idx=tr_idx,
                    trtemask=trtemask, labels=labels, mask=mask,
                    dp_fea0=dp_fea0, edge_pos=support[0], train_mat01=train_mat01,
                    mat01_tr_te=mats_intra_inter, weight_decay=weight_decay,
                    enable_contrastive_loss=args.enable_contrastive_loss,
                    enable_generative_loss=args.enable_generative_loss, loss_para_1=args.loss_para_1,
                    loss_para_2=args.loss_para_2,
                    combine_para=args.combine_para)

sess = tf.Session()

GCNmodel_loss = tf.summary.scalar('loss', GCNmodel.loss)
GCNmodel_accuracy = tf.summary.scalar('accuracy', GCNmodel.accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(
    'logs_' + '{0:%Y_%m_%d/}'.format(datetime.now()) + '{0:%Y_%m_%d_%H_%M_%S/}'.format(datetime.now()),
    sess.graph)

sess.run(tf.global_variables_initializer())

test_accs = []
train_losses = []
train_accs = []
test_losses = []
val_accs = []
val_losses = []
real_test_accs = []
real_test_loss_acc = []

support_augmented = support
features_augmented = features

for epoch in range(epochs):
    feed_dict = construct_feed_dict_1(support_augmented, features_augmented, y_train, train_mask, placeholders, mask,
                                      labels)
    feed_dict.update({placeholders['dropout']: dropout_all})
    outs = sess.run([GCNmodel.opt_op, GCNmodel_loss, GCNmodel_accuracy, GCNmodel.outputs], feed_dict=feed_dict)

    if enable_spatial_augmentation and (epoch % 100 == 0):
        support_augmented = spatial_augmentation(outs[3], support_1, graph_data.sp_nei,
                                                 args.spatial_para)
        support_augmented = sp.coo_matrix(support_augmented)
        support_augmented = sparse_to_tuple(support_augmented)

    if enable_spectral_augmentation and (epoch % 100 == 0):
        features_augmented = spectral_level_graph_augmentation(dataset_name, adj, features_1)
        features_augmented = sp.coo_matrix(features_augmented)
        features_augmented = sparse_to_tuple(features_augmented)


    writer.add_summary(outs[1], epoch)
    writer.add_summary(outs[2], epoch)

    if epoch == (epochs - 1):
        feed_dict.update({mask: test_mask})
        feed_dict.update({labels: y_test})
        feed_dict.update({placeholders['dropout']: 0})
        outs_val = sess.run([GCNmodel.loss, GCNmodel.accuracy], feed_dict=feed_dict)

        print("Epoch:", '%04d' % (epoch + 1),
              "test_accuracy=", "{:.5f}".format(outs_val[1]),
              "test_loss=", "{:.5f}".format(outs_val[0]))
        train_accs.append(outs[2])
        test_accs.append(outs_val[1])
        test_losses.append(outs_val[0])
        train_losses.append(outs[1])

print("test result:", np.max(test_accs))


pixel_wise_pred = np.argmax(outs[3], axis=1)

train_mask = np.matlib.reshape(np.array(graph_data.trmask, dtype='bool'),
                               [np.shape(graph_data.trmask)[0], 1])
test_mask = np.matlib.reshape(np.array(graph_data.temask, dtype='bool'),
                              [np.shape(graph_data.trmask)[0], 1])

pred_mat = AssignLabels(HSI_data['useful_sp_lab'], np.argmax(y_train, axis=1), pixel_wise_pred, train_mask, test_mask)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

scio.savemat(args.output_dir + '/pred_mat_' + str(re_do_times) + '.mat', {'pred_mat': pred_mat})
stat_res = GetExcelData(HSI_data[img_gt], pred_mat, HSI_data['trpos'])
scio.savemat(args.output_dir + '/stat_res_' + str(re_do_times) + '.mat', {'stat_res': stat_res})

print('*' * 20)
print('OA:%f' % stat_res[num_classes])
print('AA:%f' % stat_res[num_classes + 1])
print('Kappa:%f' % stat_res[num_classes + 2])
print('*' * 20)

time_end = time.time()
print('start time:%f' % time_start)
print('end time:%f' % time_end)
print('total time:%f' % (time_end - time_start))

txt_result = ""
for ids, contents in enumerate(stat_res):
    txt_result = txt_result + '%f ' % contents

with open(txt_filename, "a") as file:
    file.write(txt_result + '%f' % (time_end - time_start) + '\n')
