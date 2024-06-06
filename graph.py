import numpy as np

class Graph(object):

    def __init__(self, node_num, edge_num):
        self.node_num = node_num
        self.edge_num = edge_num
        self.adj_list = np.zeros(edge_num, dtype=np.int32) - 1
        self.adj_idx = np.zeros(node_num + 1,
                                dtype=np.int32)
        self.adj_wgt = np.zeros(edge_num,
                                dtype=np.float32)
        self.node_wgt = np.zeros(node_num, dtype=np.int32)
        self.cmap = np.zeros(node_num, dtype=np.int32) - 1

        self.degree = np.zeros(node_num, dtype=np.float32)
        self.A = None
        self.C = None

        self.coarser = None
        self.finer = None

    def resize_adj(self, edge_num):
        self.adj_list = np.resize(self.adj_list, edge_num)
        self.adj_wgt = np.resize(self.adj_wgt, edge_num)

    def get_neighs(self, idx):
        istart = self.adj_idx[idx]
        iend = self.adj_idx[idx + 1]
        return self.adj_list[istart:iend]

    def get_neigh_edge_wgts(self, idx):
        istart = self.adj_idx[idx]
        iend = self.adj_idx[idx + 1]
        return self.adj_wgt[istart:iend]
