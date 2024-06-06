from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from utils import *
from models import HGCN
from coarsen import *
import copy
from BuildSPInst_A import *


def HGCN_Model(placeholders, paras, per_class):
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', paras['dataset'], 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('model', 'hgcn', 'Model string.')  # 'hgcn', 'gcn', 'gcn_cheby', 'dense'

    flags.DEFINE_integer('seed1', 123, 'random seed for numpy.')
    flags.DEFINE_integer('seed2', 123, 'random seed for tf.')
    flags.DEFINE_integer('hidden', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('node_wgt_embed_dim', 5, 'Number of units for node weight embedding.')

    flags.DEFINE_float('weight_decay', paras['weight_decay'], 'Weight for L2 loss on embedding matrix.')

    flags.DEFINE_integer('coarsen_level', 4, 'Maximum coarsen level.')
    flags.DEFINE_integer('max_node_wgt', 50, 'Maximum node_wgt to avoid super-node being too large.')
    flags.DEFINE_integer('channel_num', 4, 'Number of channels')
    flags.DEFINE_integer('r', 1, 'This is the i-th time to retrain the model.')
    flags.DEFINE_string('f', 'txt', 'This is the file name of txt.')
    flags.DEFINE_string('d', 'dtxt', 'This is the file name of txt.')
    flags.DEFINE_string('t', 'ttxt', 'This is the file name of txt.')
    flags.DEFINE_string('enable_spatial_augmentation', 'enable_spatial_augmentation', 'This is the file name of txt.')
    flags.DEFINE_string('enable_spectral_augmentation', 'enable_spectral_augmentation', 'This is the file name of txt.')
    flags.DEFINE_string('spatial_para', 'spatial_para', 'This is the file name of txt.')
    flags.DEFINE_string('spectral_para', 'spectral_para', 'This is the file name of txt.')
    flags.DEFINE_string('enable_contrastive_loss', 'enable_contrastive_loss', 'This is the file name of txt.')
    flags.DEFINE_string('enable_generative_loss', 'enable_generative_loss', 'This is the file name of txt.')
    flags.DEFINE_string('per_class', 'per_class', 'This is the file name of txt.')

    flags.DEFINE_string('loss_para_1', 'loss_para', 'This is the file name of txt.')
    flags.DEFINE_string('loss_para_2', 'loss_para', 'This is the file name of txt.')
    flags.DEFINE_string('combine_para', 'combine_para', 'This is the file name of txt.')

    seed1 = FLAGS.seed1
    seed2 = FLAGS.seed2
    np.random.seed(seed1)
    tf.set_random_seed(seed2)

    print(FLAGS.dataset)
    img_gyh = FLAGS.dataset + '_gyh'
    img_gt = FLAGS.dataset + '_gt'
    HSI_data = load_HSI_data(FLAGS.dataset, per_class)
    graph_data = GetInst_A(HSI_data['useful_sp_lab'], HSI_data[img_gyh], HSI_data[img_gt], HSI_data['trpos'])
    features_sp = np.array(graph_data.sp_mean, dtype='float32')
    y_train = np.array(graph_data.sp_label, dtype='float32')
    y_val = y_train
    y_test = y_train
    train_mask = np.matlib.reshape(np.array(graph_data.trmask, dtype='bool'),
                                   [np.shape(graph_data.trmask)[0], 1])
    test_mask = np.matlib.reshape(np.array(graph_data.temask, dtype='bool'),
                                  [np.shape(graph_data.trmask)[0], 1])

    adj = graph_data.sp_A
    adj = np.array(adj, dtype='float32')
    adj = np.squeeze(adj)
    support = np.array(graph_data.CalSupport(adj), dtype='float32')

    model_func = HGCN

    graph, mapping = read_graph_from_adj(graph_data)
    print('total nodes:', graph.node_num)

    original_graph = graph
    transfer_list = []
    adj_list = [copy.copy(graph.A)]
    node_wgt_list = [copy.copy(graph.node_wgt)]
    for i in range(FLAGS.coarsen_level):
        match, coarse_graph_size = generate_hybrid_matching(FLAGS.max_node_wgt, graph)
        coarse_graph = create_coarse_graph(graph, match, coarse_graph_size)
        transfer_list.append(copy.copy(graph.C))
        graph = coarse_graph
        adj_list.append(copy.copy(graph.A))
        node_wgt_list.append(copy.copy(graph.node_wgt))
        print('There are %d nodes in the %d coarsened graph' % (graph.node_num, i + 1))

    print("\n")
    print('layer_index ', 1)
    print('input shape:   ', np.shape(features_sp)[0])

    for i in range(len(adj_list)):
        adj_list[i] = [preprocess_adj(adj_list[i])]

    return model_func(placeholders, input_dim=np.shape(features_sp)[1], logging=True, transfer_list=transfer_list,
                      adj_list=adj_list, node_wgt_list=node_wgt_list)


if __name__ == "__main__":
    HGCNModel = HGCN_Model(placeholders, paras)
