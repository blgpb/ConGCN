from collections import defaultdict
from graph import Graph
import numpy as np
from utils import cmap2C

def normalized_adj_wgt(graph):
    adj_wgt = graph.adj_wgt
    adj_idx = graph.adj_idx
    norm_wgt = np.zeros(adj_wgt.shape, dtype=np.float32)
    degree = graph.degree
    for i in range(graph.node_num):
        for j in range(adj_idx[i], adj_idx[i + 1]):
            neigh = graph.adj_list[j]
            norm_wgt[j] = adj_wgt[neigh] / np.sqrt(degree[i] * degree[neigh])
    return norm_wgt

def generate_hybrid_matching(max_node_wgt, graph):
    node_num = graph.node_num
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt
    node_wgt = graph.node_wgt
    cmap = graph.cmap
    norm_adj_wgt = normalized_adj_wgt(graph)
    groups = []
    matched = [False] * node_num

    jaccard_idx_preprocess(graph, matched, groups)

    degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]

    sorted_idx = np.argsort(degree)
    for idx in sorted_idx:
        if matched[idx]:
            continue
        max_idx = idx
        max_wgt = -1
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j]
            if neigh == idx:
                continue
            curr_wgt = norm_adj_wgt[j]
            if ((not matched[neigh]) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
                max_idx = neigh
                max_wgt = curr_wgt

        matched[idx] = matched[max_idx] = True
        if idx == max_idx:
            groups.append([idx])
        else:
            groups.append([idx, max_idx])
    coarse_graph_size = 0
    for idx in range(len(groups)):
        for ele in groups[idx]:
            cmap[ele] = coarse_graph_size
        coarse_graph_size += 1
    return (groups, coarse_graph_size)


def jaccard_idx_preprocess(graph, matched, groups):
    neighs2node = defaultdict(list)
    for i in range(graph.node_num):
        neighs = str(sorted(graph.get_neighs(i)))
        neighs2node[neighs].append(i)
    for key in neighs2node.keys():
        g = neighs2node[key]
        if len(g) > 1:
            for node in g:
                matched[node] = True
            groups.append(g)
    return


def create_coarse_graph(graph, groups, coarse_graph_size):
    coarse_graph = Graph(coarse_graph_size, graph.edge_num)
    coarse_graph.finer = graph
    graph.coarser = coarse_graph
    cmap = graph.cmap
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt
    node_wgt = graph.node_wgt

    coarse_adj_list = coarse_graph.adj_list
    coarse_adj_idx = coarse_graph.adj_idx
    coarse_adj_wgt = coarse_graph.adj_wgt
    coarse_node_wgt = coarse_graph.node_wgt
    coarse_degree = coarse_graph.degree

    coarse_adj_idx[0] = 0
    nedges = 0
    for idx in range(len(groups)):
        coarse_node_idx = idx
        neigh_dict = dict()
        group = groups[idx]
        for i in range(len(group)):
            merged_node = group[i]
            if (i == 0):
                coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
            else:
                coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]

            istart = adj_idx[merged_node]
            iend = adj_idx[merged_node + 1]
            for j in range(istart, iend):
                k = cmap[adj_list[
                    j]]
                if k not in neigh_dict:
                    coarse_adj_list[nedges] = k
                    coarse_adj_wgt[nedges] = adj_wgt[j]
                    neigh_dict[k] = nedges
                    nedges += 1
                else:
                    coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]

                coarse_degree[coarse_node_idx] += adj_wgt[j]

        coarse_node_idx += 1
        coarse_adj_idx[coarse_node_idx] = nedges

    coarse_graph.edge_num = nedges

    coarse_graph.resize_adj(nedges)
    C = cmap2C(cmap)
    graph.C = C
    coarse_graph.A = C.transpose().dot(graph.A).dot(C)
    return coarse_graph
