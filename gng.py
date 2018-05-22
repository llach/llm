# -*- coding: utf-8 -*-

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import uuid
import sys
import math

# calculate euclidean distance for vectors x and y
def euclidean_dist(x, y, debug=False):

    # check length
    if(len(x) is not len(y)):
        raise Exception('Vector length must align!')

    dist = 0

    # calculate euclidean distance
    for xi, yi in zip(x, y):
        dist += (xi - yi)**2

    dist = math.sqrt(dist)

    if debug:
        print('distance: %f' % dist)

    return dist 

class Node:

    def __init__(self, id, pos):

        # UUID
        self.id = id
        
        # nodes position in data space
        self.pos = pos

        # nodes neighbors
        self.neighbors = set([])

        # edges from node to neighbors
        self.edges = set([])


class Edge:

    def __init__(self, n0, n1):
        
        # nodes connected by this edge
        self.n0 = n0
        self.n1 = n1

        # edge age
        self.age = 0

        # line plot item for visualization
        self.line = gl.GLLinePlotItem(pos=np.array([n0.pos, n1.pos]), color=(.0, 1., 1., 1.), antialias=True)
        
        # add nodes in neighborhood lists
        n0.neighbors.add(n1)
        n1.neighbors.add(n0)
        
        # also add ourselves to node edge lists
        n0.edges.add(self)
        n1.edges.add(self)

    def prepare_removal(self):
        
        # delete neighborhood of edges nodes
        self.n0.neighbors.remove(self.n1)
        self.n1.neighbors.remove(self.n0)

        # delete ourselves from nodes edge lists
        self.n0.edges.remove(self)
        self.n1.edges.remove(self)


class GNG:

    def __init__(self, eta_n=0.1, eta_c=0.05, age_max=10, alpha=0, beta=0,
                 node_intervall=10, dist=euclidean_dist, freq=50, debug=False):

        self.name = 'GNG - Growing Neural Gas'

        # set and check network parameter
        if eta_n <= eta_c:
            raise Exception('eta_n has to be larger than eta_c!')
        
        self.eta_n = eta_n
        self.eta_c = eta_c

        # TODO sanity checks for alpha & beta?
        self.age_max = age_max
        self.alpha = alpha
        self.beta = beta
        self.node_intervall = node_intervall

        # init node and edge sets
        self.nodes = set([])
        self.edges = set([])

        # save iteration we are in
        self.timestep = 0

        # store number of nodes present in network
        self.node_count = 0

        # save reference to training data
        self.data = None

        # distance function to use
        self.dist = dist

        # data for network plotting
        self.node_positions = None
        self.node_colors = None
        self.node_sizes = None

        # True for debug output
        self.debug = debug
        
        # prepare QTGraph
        self.app = QtGui.QApplication([])
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 3
        self.w.show()
        self.w.setWindowTitle(self.name)

        g = gl.GLGridItem()
        self.w.addItem(g)

        # create timer that calls train function
        self.t = QtCore.QTimer()
        self.t.timeout.connect(self.train)

        # QT update frequency
        self.freq = 50

    # prepare data for plotting
    def draw(self):
        
        # create position array
        self.node_positions = np.array([n.pos for n in self.nodes])

        # create color array
        self.node_colors = np.array(self.node_count*[[0, 1, 0, 0.9]])

        # create size array
        self.node_sizes = np.array(self.node_count*[0.025])

        # update plot item
        self.node_plot.setData(pos=self.node_positions, size=self.node_sizes, color=self.node_colors)
        
    # simple node adding function
    def add_node(self, position):
        
        # create and add node to network
        n = Node(str(uuid.uuid4())[:8], position)
        self.nodes.add(n)
        
        # increase node count
        self.node_count += 1

        return n

    def add_edge(self, n0, n1):

        # create edge
        e = Edge(n0, n1)
        self.edges.add(e)

        # add edge to plot
        self.w.addItem(e.line)

        return e

    def remove_edge(self, e):

        # get edge ready for safe removal
        e.prepare_removal()

        # remove edge from set
        self.edges.remove(e)

        # remove edge from plot widget
        self.w.removeItem(e)

    def adapt(self, node):
        pass

    def edge_update(self):
        pass

    def node_update(self):
        pass

    def find_nearest(self, stimulus):
        
        nearest = None
        second = None

        # search through all nodes in network
        for node in self.nodes:
            
            # calculcate node dist only once
            node_dist = self.dist(node.pos, stimulus, self.debug) 

            # first iteration -> no search required
            if nearest is None:
                nearest = node
                continue
            # second iteration -> is second node new nearest or second nearest
            elif second is None:
                if node_dist <= self.dist(nearest.pos, stimulus, self.debug):
                    second = nearest
                    nearest = node
                else:
                    second = node
                continue

            # node is further away from stimulus then second
            if node_dist >= self.dist(second.pos, stimulus, self.debug):
                continue
            # stimulus between nearest and second
            elif node_dist >= self.dist(nearest.pos, stimulus, self.debug):
                second = node
                continue
            # stimulus nearer than nearest
            else:
                second = nearst
                nearest = node
            

        if self.debug:
            print('nearest: %s; second: %s' % (nearest.id, second.id))

        # some small sanity check
        if nearest == second:
            raise Exception('nearst cannot be second nearest!')

        return nearest, second 


    # takes care of training and visualization
    def run(self, data):

        # store data reference
        self.data = data
        
        # number of datapoints
        n = data.shape[0]

        # create color and size arrays
        color = np.array(n*[[1, 0, 0, 0.7]])
        size = np.array(n*[0.02])

        # plot data distribution
        self.dist_plot = gl.GLScatterPlotItem(pos=data, size=size, color=color, pxMode=False)
        self.w.addItem(self.dist_plot)
        
        # plot nodes
        self.node_plot = gl.GLScatterPlotItem(pos=self.node_positions, size=self.node_sizes, color=self.node_colors, pxMode=False)
        self.w.addItem(self.node_plot)

        # add initial nodes
        for x in data[:2]:
            self.add_node(x)

        # increment timestep after adding nodes for correct indexing
        self.timestep += 2

        # plot network
        self.draw()

        # start timer, therefore training
        self.t.start(self.freq)

    
    def train(self):
        
        # TODO randomly chose datapoints, should not depent on iteration step for GNG!
        # only train network if we have datapoints left
        if(self.timestep >= self.data.shape[0]):
            return

        # select datapoint based on current timestep
        stimulus = self.data[self.timestep-1]

        # find n, s
        n, s = self.find_nearest(stimulus)

        # plot network
        self.draw()
        
        # increment timestep
        self.timestep += 1

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        gng = GNG(debug=True)

        data = np.random.rand(100, 3)

        gng.run(data)
        QtGui.QApplication.instance().exec_()


