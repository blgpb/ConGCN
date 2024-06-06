import numpy as np

def spatial_augmentation(H, support, nei, p):

    for ids, node in enumerate(nei):
        for i in node:
            d = np.sum(np.power(H[ids] - H[i - 1], 2))
            min_d = np.minimum(np.exp(0.2 * np.sum(np.power(H[ids] - H[i - 1], 2), axis=0)), 2)
            support[ids, i - 1] = support[ids, i - 1] - p * np.sign(d - 0.5) * min_d

    return support

