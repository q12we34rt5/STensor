import numpy as np


def reduce_dim(v: np.ndarray, shape: tuple):
    assert len(shape) <= len(v.shape)
    ext_shape = (1,) * (len(v.shape) - len(shape)) + shape
    sum_axis = tuple(i for i in range(len(ext_shape)) if ext_shape[i] == 1)
    return v.sum(axis=sum_axis).reshape(shape)
