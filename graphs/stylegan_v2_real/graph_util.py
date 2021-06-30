from . import constants
import numpy as np


def z_sample(batch_size, seed=0, dim_z=constants.DIM_Z):
    rnd = np.random.RandomState(seed)
    zs = rnd.randn(batch_size, dim_z)
    return zs


def w_sample(batch_size, seed=0, dim_z=constants.DIM_Z):
    rnd = np.random.RandomState(seed)
    ws = rnd.uniform(low=-1, high=2, size=(batch_size, dim_z))
    return ws

def graph_input(graph, num_samples, seed=0, **kwargs):
    ''' creates z inputs for graph '''
    zs = z_sample(num_samples, seed, graph.dim_z)
    return {'z': zs}