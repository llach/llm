from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

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
        self.eta_out = kwargs.get('eta_out', 0.1)
        self.eta_a = kwargs.get('eta_a', 0.1)
        self.y = kwargs.get('interpolation_function',simple_vq)
        self.sigmas = []

        # QtGui.QApplication.setGraphicsSystem('raster')
        app = QtGui.QApplication([])

        win = pg.GraphicsWindow(title=self.name)
        win.resize(1000, 600)
        win.setWindowTitle(self.name)

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        self.node_plot = None
        self.edge_plot = None

        timer = QtCore.QTimer()
        timer.timeout.connect(self.train)
        timer.start(50)

    def init(self, data, targets):
        self.data = data
        self.targets = targets

        self.mapping_method.data = self.data
        new_nodes, stimulus_idxs = self.mapping_method.prepare()

        for node, idx in zip(new_nodes, stimulus_idxs):
            node.init_llm_params(self.targets[idx])

        self.node_plot = self.win.addPlot()

    def draw(self):

        # create position array
        if self.mapping_method.data_dim is 1:
            node_positions = np.array([[n.pos, 0] for n in self.nodes])
        else:
            node_positions = np.array([n.pos for n in self.nodes])

        # create color array
        node_colors = np.array(self.node_count * [[0, 1, 0, 0.9]])

        # create size array
        node_sizes = np.array(self.node_count * [0.025])

        # update plot item
        self.node_plot.setData(pos=self.node_positions, size=self.node_sizes, color=self.node_colors)

        # update all edge positions
        for e in self.edges:
            e.update_plot_item()

    def run(self):
        print('starting llm')
        QtGui.QApplication.instance().exec_()

    def train(self, max_it=10000):

        #TODO add timestep t
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

        print(n.pos)


if __name__ == '__main__':
    llm = LLM()
    data_in = np.linspace(start=0, stop=10*math.pi, num=400) # gng node max==100, 4 points per node
    targets = [math.sin(x) for x in data_in] # targets for supervised w_out part
    llm.init(data_in, targets)
    llm.train()