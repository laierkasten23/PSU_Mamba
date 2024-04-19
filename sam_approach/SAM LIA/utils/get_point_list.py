import numpy as np


def get_point_list(points):
    # return np.concatenate(
    #    list(map(
    #        lambda x : np.reshape(np.array(x),(len(x),2)),
    #        points
    #    ))
    # )

    return np.concatenate(points)
