import numpy as np
import math

from .util.euclidean_dist import euclidean_dist
from topological_map import Node
from gng import GNG
from itm import ITM

def find_nearest_idx(x, w_in):
    dists = [euclidean_dist(x, w.pos) for w in w_in]
    idx_closest = np.argmin(dists)
    return idx_closest, w_in[idx_closest]

def simple_vq(x, w_in, w_out):
    idx, _ = find_nearest_idx(x, w_in)
    return w_out[idx]

def soft_max(x, w_in, w_out, sigmas, neighborhood_size=1):
    idx, n = find_nearest_idx(x, w_in)
    neighbors = n.get_grid_neighbors(neighborhood_size)
    N = len(neighbors)



def std_llm(x, w_in, w_out, A):
    idx, _ = find_nearest_idx(x, w_in)
    w_o = w_in[idx]
    return w_o + A[idx]*(np.subtract(x, w_in[idx]))


class LLM(object):

    def __init__(self, *args, **kwargs):
        self.name="LLM - Local Linear Map"
        self.mapping_method = kwargs.get('mapping', GNG())
        self.interpolation_str = kwargs.get('interpolation_function','VQ')
        self.sigmas = []
        self.eta_out = kwargs.get('eta_out', 0.1)
        self.eta_a = kwargs.get('eta_a', 0.1)
        self.y = None # interpolation function
        self.set_interpolation_function(self.interpolation_str)

    # sets interpolation function that is to be used, defaults to VQ
    def set_interpolation_function(self, str_fun):
        if str_fun == 'soft_max':
            self.y = soft_max
        elif str_fun == 'std_llm':
            self.y = std_llm
        else:
            self.y = simple_vq


    def init(self, data, targets):
        self.data = data
        self.targets = targets

    def run(self):

        self.mapping_method.llm_done = True


if __name__=='__main__':
    llm = LLM()
    data_in = np.linspace(start=0, stop=10*math.pi, num=400) # gng node max==100, 4 points per node
    targets = [math.sin(x) for x in data_in] # targets for supervised w_out part
    llm.init(data_in, targets)
    llm.run()