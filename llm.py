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
        self.app = QtGui.QApplication([])

        self.win = pg.GraphicsWindow(title=self.name)
        self.win.resize(1000, 600)
        self.win.setWindowTitle(self.name)

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        self.node_plot = None
        self.edge_plots = []
        self.wc_plot = None
        self.wc_edges = []

        self.node_view = self.win.addPlot()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.train)


    def init(self, data, targets):
        self.data = data
        self.targets = targets

        self.mapping_method.data = self.data
        new_nodes, stimulus_idxs = self.mapping_method.prepare()

        for node, idx in zip(new_nodes, stimulus_idxs):
            node.init_llm_params(self.targets[idx])

        self.node_plot = pg.ScatterPlotItem( pen=pg.mkPen('r'))
        self.node_view.addItem(self.node_plot)

        self.wc_plot = pg.ScatterPlotItem(pen=pg.mkPen('g'))
        self.node_view.addItem(self.wc_plot)


    def draw(self):

        # node positions
        pos = np.array([np.array([n.pos, 0]) for n in self.mapping_method.nodes])
        self.node_plot.setData(pos=pos)

        # wc plot
        wc_pos = np.array([np.array([n.pos, n.w_out]) for n in self.mapping_method.nodes])
        self.wc_plot.setData(pos=wc_pos)

        # remove wc edges
        # for wep in self.wc_edges:
        #     self.node_view.removeItem(wep)
        #
        # # build new wc edges
        # for i in range(self.mapping_method.node_count - 1):
        #     ed = pg.PlotCurveItem()
        #     ed.setData(np.array([wc_pos[i][0], wc_pos[i+1][0]]), np.array([wc_pos[i][1], wc_pos[i+1][1  ]]))
        #     self.wc_edges.append(ed)
        #     self.node_view.addItem(ed)

        # remove all edges
        for ep in self.edge_plots:
            self.node_view.removeItem(ep)

        # update all edge and plot them
        for e in self.mapping_method.edges:
            ed = pg.PlotCurveItem()
            e.update_plot_item()
            ed.setData(np.array([ e.n0.pos, e.n1.pos ]), np.array([ 0, 0 ])  )
            self.edge_plots.append(ed)
            self.node_view.addItem(ed)

    def run(self):
        print('starting llm')
        self.timer.start(50)
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
        self.draw()


if __name__ == '__main__':
    llm = LLM()
    data_in = np.linspace(start=0, stop=3*math.pi, num=400) # gng node max==100, 4 points per node
    targets = [math.sin(x) for x in data_in] # targets for supervised w_out part
    llm.init(data_in, targets)
    llm.run()