from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

import math

from util.euclidean_dist import euclidean_dist
from gng import GNG
from itm import ITM


def simple_vq(x, node, neighborhood_size=2,neighboring='grid_space', whole_network=False, network=None):
    return node.w_out


def soft_max(x, node, neighborhood_size=2,neighboring='grid_space', whole_network=False, network=None):
    neighbors = get_neighborhood(node=node, radius=neighborhood_size, neighboring=neighboring,whole_network=whole_network, network=network)
    N = len(neighbors)
    wc = [np.exp(-(euclidean_dist(x, n.pos)**2)/(n.sigma**2)) for n in neighbors]
    gc = [w/N for w in wc]
    return np.sum([g*n.w_out for g,n in zip(gc,neighbors)])


def std_llm(x, node, neighborhood_size=2,neighboring='grid_space', whole_network=False, network=None):
    if np.isscalar(node.A):
        node.A = [node.A]
    diff = np.subtract(x, node.pos)
    if np.isscalar(diff):
        diff = [diff]
    return np.add(node.w_out, np.matmul(node.A, diff))


def soft_max_llm(x,node,neighborhood_size=2, neighboring='grid_space', whole_network=False, network=None):
    neighbors = get_neighborhood(node=node, radius=neighborhood_size, neighboring=neighboring,whole_network=whole_network, network=network)

    N = len(neighbors)
    wc = [np.exp(-(euclidean_dist(x, n.pos)**2)/(n.sigma**2)) for n in neighbors]
    gc = [w/N for w in wc]
    return np.sum([g*std_llm(x,n) for g,n in zip(gc,neighbors)])


def get_neighborhood(node, radius=2, neighboring='grid_space', whole_network=False, network=None):
    if neighboring is 'input_space':
        if whole_network:
            if network is None:
                raise Exception("no network given")
            else:
                return network
        else:
            neighbors = set([node])
            dists = [euclidean_dist(node.pos, neighbor.pos) for neighbor in node.neighbors]
            closest_idx = np.argsort(dists)
            self_neighbors_array = np.array(list(node.neighbors))
            xx = len(closest_idx)
            for i in range(min(radius, len(closest_idx))):
                neighbors.add(self_neighbors_array[closest_idx[i]])

            return neighbors
    else:
        return node.get_grid_neighbors(radius)


class LLM(object):

    def __init__(self, *args, **kwargs):
        self.name="LLM - Local Linear Map"
        self.mapping_method = kwargs.get('mapping', GNG())
        self.eta_out = kwargs.get('eta_out', 0.1)
        self.eta_a = kwargs.get('eta_a', 0.1)
        self.eta_sigma = kwargs.get('eta_sigma', 0.1)
        self.y = kwargs.get('interpolation_function', soft_max_llm)

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

        delta_win, x, n, s, new_node, stimulus_idx = self.mapping_method.train()
        delta_win = [delta_win[0][1]]
        if new_node is not None:
            new_node.init_llm_params(self.targets[stimulus_idx])

        # adapt w_out
        y_t = self.targets[stimulus_idx]
        y_x = self.y(x, n, neighborhood_size=4, neighboring='input_space', whole_network=True, network=self.mapping_method.nodes)
        y_error = np.subtract(y_t, y_x)

        if np.isscalar(y_error):
            y_error = [y_error]

        A_w = np.matmul(n.A, delta_win)
        delta_wout = self.eta_out * np.add(y_error, A_w)
        n.w_out = np.add(n.w_out, delta_wout)
        # adapt sigma if self.y is soft-max*
        if self.y is soft_max or self.y is soft_max_llm:
            n.sigma += self.eta_sigma * np.subtract(euclidean_dist(x, n.pos), n.sigma)

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