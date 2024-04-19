import numpy as np


def get_label_list(ll):
    return np.repeat(range(len(ll)), list(map(len, ll)))
