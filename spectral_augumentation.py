import random
import numpy as np

def sample_random_vector_m(exchange_prob):

    exchange_mask = np.zeros(shape=(exchange_prob.shape[0]), dtype=bool)
    for i in range(exchange_prob.shape[0]):
        exchange_mask[i] = bool(np.random.binomial(1, exchange_prob[i]))

    return exchange_mask


def spectral_level_graph_augmentation(dataset_name, A, X):

    exchange_prob = np.load(dataset_name + '_mutual_information.npy')


    adj = A.copy()

    adj_squeeze = adj.reshape((-1,))
    s = np.argsort(-adj_squeeze)

    have_changed_nodes = []

    for i in range(0, s.shape[0], 2):

        pos = s[i]
        y = int(pos % adj.shape[0])
        x = int((pos - y) / adj.shape[1])
        adjacency_value = adj[x][y]
        if adjacency_value > 0:

            if x not in have_changed_nodes and y not in have_changed_nodes:
                exchange_mask = sample_random_vector_m(exchange_prob)

                temp_tensor_A = X[x].copy()

                X[x, exchange_mask] = X[y, exchange_mask]
                X[y, exchange_mask] = temp_tensor_A[exchange_mask]

                have_changed_nodes.append(x)
                have_changed_nodes.append(y)
        else:
            break

    return X


def A_Maxpooling(adj):

    adj_Maxpooling = np.zeros(shape=(adj.shape[0]), dtype=np.float)
    for i in range(adj.shape[0]):
        adj_Maxpooling[i] = np.max(adj[i, :])
    return adj_Maxpooling


def A_Accumulate(adj):

    adj_Accumulate = np.zeros(shape=(adj.shape[0]), dtype=np.float)
    for i in range(adj.shape[0]):
        adj_Accumulate[i] = np.sum(adj[i, :])
    return adj_Accumulate


def H_Maxpooling(H):

    H_Maxpooling = np.zeros(shape=(H.shape[0]), dtype=np.float)
    for i in range(H.shape[0]):
        H_Maxpooling[i] = np.max(H[i, :])
    return H_Maxpooling


def H_Accumulate(H):

    H_Accumulate = np.zeros(shape=(H.shape[0]), dtype=np.float)
    for i in range(H.shape[0]):
        H_Accumulate[i] = np.sum(H[i, :])
    return H_Accumulate


def feature_drop_weights_dense(x, adj):
    x = np.abs(x)

    w = np.dot(x.T, A_Accumulate(adj))

    w = np.log(w)
    s = (np.max(w) - w) / (np.max(w) - np.mean(w))

    return s


def drop_feature_weighted_2(dataset_name, x, graph_data, adj, p: float, threshold: float = 0.7,
                            is_limit_change_times=True):
    nei = graph_data.sp_nei

    drop_prob = np.load(dataset_name + '_mutual_information.npy')
    drop_prob = 1 - drop_prob
    drop_prob = drop_prob * 0.2

    drop_mask = np.zeros(shape=(x.shape[1]), dtype=bool)
    for i in range(x.shape[1]):
        drop_mask[i] = bool(np.random.binomial(1, drop_prob[i]))

    have_changed_nodes = []
    edges_count = 0
    for ids, node in enumerate(nei):
        nodes_list = []
        nodes_list.append(ids)
        for i in node:
            nodes_list.append(i - 1)

        if is_limit_change_times:
            for nod in have_changed_nodes:
                if nod in nodes_list:
                    nodes_list.remove(nod)

        if len(nodes_list) % 2 == 0:
            random.shuffle(nodes_list)
            A_list = nodes_list[0:int(len(nodes_list) / 2)]
            B_list = nodes_list[int(len(nodes_list) / 2):]
        else:
            drop_one_node_index = random.randint(0, len(nodes_list) - 1)
            nodes_list.pop(drop_one_node_index)

            random.shuffle(nodes_list)
            A_list = nodes_list[0:int(len(nodes_list) / 2)]
            B_list = nodes_list[int(len(nodes_list) / 2):]

        assert len(A_list) == len(B_list)

        for idss, node_of_A_list in enumerate(A_list):
            temp_tensor_A = x[node_of_A_list].copy()

            x[node_of_A_list, drop_mask] = x[B_list[idss], drop_mask]
            x[B_list[idss], drop_mask] = temp_tensor_A[drop_mask]
        have_changed_nodes = nodes_list

    return x


def feature_drop_weights_adaptive(x, H):
    x = np.abs(x)
    w = np.dot(x.T, H_Maxpooling(H))

    w = np.log(w)
    s = (np.max(w) - w) / (np.max(w) - np.mean(w))

    return s


def drop_feature_weighted_adaptive(x, graph_data, H, p: float, threshold: float = 0.7, is_limit_change_times=True):
    nei = graph_data.sp_nei

    w = feature_drop_weights_adaptive(x, H)
    w = w / np.mean(w) * p
    w = np.where(w < threshold, w, np.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = np.zeros(shape=(x.shape[1]), dtype=bool)
    for i in range(x.shape[1]):
        drop_mask[i] = bool(np.random.binomial(1, drop_prob[i]))

    have_changed_nodes = []
    edges_count = 0
    for ids, node in enumerate(nei):
        nodes_list = []
        nodes_list.append(ids)
        for i in node:
            nodes_list.append(i - 1)

        if is_limit_change_times:
            for nod in have_changed_nodes:
                if nod in nodes_list:
                    nodes_list.remove(nod)

        if len(nodes_list) % 2 == 0:
            random.shuffle(nodes_list)
            A_list = nodes_list[0:int(len(nodes_list) / 2)]
            B_list = nodes_list[int(len(nodes_list) / 2):]
        else:
            drop_one_node_index = random.randint(0, len(nodes_list) - 1)
            nodes_list.pop(drop_one_node_index)

            random.shuffle(nodes_list)
            A_list = nodes_list[0:int(len(nodes_list) / 2)]
            B_list = nodes_list[int(len(nodes_list) / 2):]

        assert len(A_list) == len(B_list)

        for idss, node_of_A_list in enumerate(A_list):
            temp_tensor_A = x[node_of_A_list]

            x[node_of_A_list, drop_mask] = x[B_list[idss], drop_mask]
            x[B_list[idss], drop_mask] = temp_tensor_A[drop_mask]
        have_changed_nodes = nodes_list

    return x
