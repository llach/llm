import numpy as np
import math

from util.euclidean_dist import euclidean_dist
from gng import GNG
from itm import ITM

def find_nearest_idx(x, w_in):
    dists = [euclidean_dist(x, w.pos) for w in w_in]
    idx_closest = np.argmin(dists)
    return idx_closest, w_in[idx_closest]

def simple_vq(x, node):
    return node.w_out

def soft_max(x, w_in, w_out, sigmas, neighborhood_size=1):
    idx, n = find_nearest_idx(x, w_in)
    neighbors = n.get_grid_neighbors(neighborhood_size)
    N = len(neighbors)

def std_llm(x, node):
    return np.add(node.w_out, np.matmul(node.A,(np.subtract(x, node.pos))))


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


    def train(self, max_it=10000):
        self.mapping_method.data = self.data
        new_nodes, stimulus_idxs = self.mapping_method.prepare()

        for node, idx in zip(new_nodes,stimulus_idxs):
            node.init_llm_params(self.targets[idx])

        for t in range(self.mapping_method.timestep, max_it):
            self.mapping_method.timestep = t
            delta_win, x, n, s, new_node, stimulus_idx = self.mapping_method.train()
            delta_win = [delta_win[0][1]]
            if new_node is not None:
                new_node.init_llm_params(self.targets[stimulus_idx])

            # adapt w_out
            y_t = self.targets[stimulus_idx]
            y_error = np.subtract(y_t, self.y(x, n))
            if np.isscalar(y_error):
                y_error = [y_error]
            A_w = np.matmul(n.A, delta_win)
            delta_wout = self.eta_out * np.add(y_error, A_w)
            n.w_out = np.add(n.w_out, delta_wout)

            # adapt A
            qe = np.subtract(x, n.pos)
            qe = [qe/np.linalg.norm(qe)]
            delta_A = self.eta_a * np.matmul(y_error, qe)
            n.A = np.add(n.A, delta_A)
            print("running")

if __name__ == '__main__':
    llm = LLM()
    data_in = np.linspace(start=0, stop=10*math.pi, num=400) # gng node max==100, 4 points per node
    targets = [math.sin(x) for x in data_in] # targets for supervised w_out part
    llm.init(data_in, targets)
    llm.train()