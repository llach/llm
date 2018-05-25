# -*- coding: utf-8 -*-

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import uuid

import numpy as np
import math
import random
import logging
from .util.euclidean_dist import euclidean_dist

class Node:

    def __init__(self, pos, grid_position=None):

        # UUID
        self.uuid = str(uuid.uuid4())[:8]

        # nodes position in data space
        self.pos = pos

        # nodes neighbors
        self.neighbors = set([])

        # edges from node to neighbors
        self.edges = set([])

        # initialize node error with zero
        self.error = 0

        # grid position for som
        self.grid_position = grid_position

    # edge to n is returned or exception thrown
    def get_edge_to_neighbor(self, n):

        for e in self.edges:
            if e.has_node(n):
                return e

        raise Exception('neighbor %s is unknown!' % n.uuid)

    # removes itself from the neighbor list of all nodes it has an edge to and also removes the edge from the neighboring nodes
    def prepare_removal(self):
        for e in self.edges:
            neigh = e.get_other_node(self)
            neigh.neighbors.remove(self)
            neigh.edges.remove(e)

    # for grid nodes: get neighbors within radius r
    def get_grid_neighbors(self, radius=1):

        # first, add ourselves to the set
        neighbors = set([self])

        for i in range(radius):

            # neighbors in next radius
            new_neighbors = set([])

            # collect neighbors of currently known nodes
            for n in neighbors:
                for x in n.neighbors:
                    new_neighbors.add(x)

            # add neighbors
            for x in new_neighbors:
                neighbors.add(x)

        return neighbors

class Edge:

    def __init__(self, n0, n1):

        # UUID
        self.uuid = str(uuid.uuid4())[:8]

        # nodes connected by this edge
        self.n0 = n0
        self.n1 = n1

        # edge age
        self.age = 0

        # line plot item for visualization
        self.line = gl.GLLinePlotItem(pos=np.array([self.n0.pos, self.n1.pos]),
                                      color=(.0, 1., 1., 1.), antialias=True)

        # add nodes in neighborhood lists
        n0.neighbors.add(n1)
        n1.neighbors.add(n0)

        # also add ourselves to node edge lists
        n0.edges.add(self)
        n1.edges.add(self)

    def __str__(self):
        return ('edge %s connecting %s to %s' % (self.uuid, self.n0.uuid, self.n1.uuid))

    # grants safe removal of edge from edge list
    def prepare_removal(self):

        # delete neighborhood of edges nodes
        self.n0.neighbors.remove(self.n1)
        self.n1.neighbors.remove(self.n0)

        # delete ourselves from nodes edge lists
        self.n0.edges.remove(self)
        self.n1.edges.remove(self)

    # checks whether this edge connects given node n
    def has_node(self, n):

        if self.n0 is n or self.n1 is n:
            return True
        else:
            return False

    # update line after nodes position change
    def update_plot_item(self):
        self.line.setData(pos=np.array([self.n0.pos, self.n1.pos]))

    # gives a node to the edge and returns the other node
    def get_other_node(self, n):
        if self.n0 is n:
            return self.n1
        elif self.n1 is n:
            return self.n0
        else:
            raise Exception("Node not part of the edge!")


class TopologicalMap(object):

    def __init__(self, *args, **kwargs):

        if self.name is None:
            self.name = '--unnamed--'

        # should this network also visualize itself?
        self.viz = kwargs.get('visualization', False)

        # creating logger
        logging.basicConfig()
        self.loggerlevel = kwargs.get('loggerlevel', 'WARNING')
        self.logger = logging.getLogger()
        self.logger.setLevel(self.loggerlevel)
        print("Initialized logger on level "+self.loggerlevel)

        # distance function to use
        self.dist = kwargs.get('dist', euclidean_dist)

        # QT update frequency
        self.freq = kwargs.get('freq', 30)

        # init node and edge sets
        self.nodes = set([])
        self.edges = set([])

        # save iteration we are in
        self.timestep = 0

        # store number of nodes and edges present in network
        self.node_count = 0
        self.edge_count = 0

        # minimum and maximum nodes
        self.node_min = kwargs.get('node_min', None)
        self.node_max = kwargs.get('node_max', None)

        # save reference to training data
        self.data = None
        self.data_dim = None

        # data for network plotting
        self.node_positions = None
        self.node_colors = None
        self.node_sizes = None

        # no need to prepare plotting if it is not needed
        if not self.viz:
            return

        # prepare QTGraph
        self.app = QtGui.QApplication([])
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 3
        self.w.show()
        self.w.setWindowTitle(self.name)

        gx = gl.GLGridItem(QtGui.QVector3D(1,1,1))
        gx.rotate(90, 0, 1, 0)
        gx.translate(0, .5, .5)
        gx.setSpacing(spacing=QtGui.QVector3D(.1,.1,.1))
        self.w.addItem(gx)
        gy = gl.GLGridItem(QtGui.QVector3D(1,1,1))
        gy.rotate(90, 1, 0, 0)
        gy.translate(.5, 0, .5)
        gy.setSpacing(spacing=QtGui.QVector3D(.1, .1, .1))
        self.w.addItem(gy)
        gz = gl.GLGridItem(QtGui.QVector3D(1,1,1))
        gz.translate(.5, .5, 0)
        gz.setSpacing(spacing=QtGui.QVector3D(.1, .1, .1))
        self.w.addItem(gz)

        # create timer that calls train function
        self.t = QtCore.QTimer()
        self.t.timeout.connect(self.train_wrapped)

    # prepare data for plotting
    def draw(self):

        # create position array
        self.node_positions = np.array([n.pos for n in self.nodes])

        # create color array
        self.node_colors = np.array(self.node_count * [[0, 1, 0, 0.9]])

        # create size array
        self.node_sizes = np.array(self.node_count * [0.025])

        # update plot item
        self.node_plot.setData(pos=self.node_positions, size=self.node_sizes, color=self.node_colors)

        # update all edge positions
        for e in self.edges:
            e.update_plot_item()

    def add_node(self, position, grid_position=None):

        if self.node_max is not None and self.node_count >= self.node_max:
            self.logger.info('Node Maximum reached!')
            return

        # create and add node to network
        n = Node(position, grid_position)
        self.nodes.add(n)

        # increase node count
        self.node_count += 1

        self.logger.debug('created node %s at %s' % (n.uuid, str(n.pos)))

        return n

    def add_edge(self, n0, n1):
        # check whether nodes are already connected
        if n0 in n1.neighbors: # or n1 in n0.neighbors:
            return
        # create edge
        e = Edge(n0, n1)
        self.edges.add(e)

        self.edge_count += 1

        self.logger.debug('created edge %s' % e.uuid)

        # add edge to plot
        if self.viz:
            self.w.addItem(e.line)

        return e

    def remove_node(self, node):

        if self.node_min is not None and self.node_count <= self.node_min:
            self.logger.info('Node Minimum reached!')
            return

        # sanity check whether node can be removed
        if len(node.neighbors) > 0 or len(node.edges) > 0:
            self.logger.debug("node removed that had neighbors or edges")

        self.node_count -= 1

        self.logger.debug('removing node %s' % node.uuid)

        node.prepare_removal()
        self.edges -= node.edges

        self.nodes.remove(node)

    def remove_edge(self, e):

        self.logger.debug('removing edge %s' % e.uuid)

        # get edge ready for safe removal
        e.prepare_removal()

        self.edge_count -= 1

        # remove edge from set
        self.edges.remove(e)

        # remove edge from plot widget
        if self.viz:
            self.w.removeItem(e.line)

    def find_nearest(self, stimulus):

        nearest = None
        second = None

        # search through all nodes in network
        for node in self.nodes:

            # calculcate node dist only once
            node_dist = self.dist(node.pos, stimulus)

            # first iteration -> no search required
            if nearest is None:
                nearest = node
                continue
            # second iteration -> is second node new nearest or second nearest
            elif second is None:
                if node_dist <= self.dist(nearest.pos, stimulus):
                    second = nearest
                    nearest = node
                else:
                    second = node
                continue

            # node is further away from stimulus than second
            if node_dist >= self.dist(second.pos, stimulus):
                continue
            # stimulus between nearest and second
            elif node_dist >= self.dist(nearest.pos, stimulus):
                second = node
                continue
            # node is closer to stimulus than nearest
            else:
                second = nearest
                nearest = node

        self.logger.debug('nearest: %s; second: %s' % (nearest.uuid, second.uuid))

        # some small sanity check
        if nearest == second:
            raise Exception('nearest cannot be second nearest!')

        if nearest == None or second == None:
            self.logger.critical('nearest or second nearest node was None!')
            raise Exception('nearest or second was none')

        return nearest, second

    # takes care of training and visualization
    def run(self, data):

        # check dimensionality of data. add zeros if needed
        if data.shape[1] < 3:

            # determine number of zeros to be added
            missing_dims = 3 - data.shape[1]

            for d in data:
                if self.data is None:
                    self.data = np.array([np.append(d, missing_dims * [0])])
                else:
                    self.data = np.vstack((self.data, np.append(d, missing_dims * [0])))
        else:
            # store data reference
            self.data = data

        # store data dimensionality
        self.data_dim = data.shape[1]

        # prepare network
        self.prepare()

        # skip viz if not needed
        if not self.viz:
            return

        # number of datapoints
        n = data.shape[0]

        # create color and size arrays
        color = np.array(n * [[1, 0, 0, 0.7]])
        size = np.array(n * [0.02])

        # plot data distribution
        self.dist_plot = gl.GLScatterPlotItem(pos=self.data, size=size, color=color, pxMode=False)
        self.w.addItem(self.dist_plot)

        # plot nodes
        self.node_plot = gl.GLScatterPlotItem(pos=self.node_positions, size=self.node_sizes, color=self.node_colors,
                                              pxMode=False)
        self.w.addItem(self.node_plot)

        # plot network
        self.draw()

        # start timer, therefore training
        self.t.start(self.freq)

        # run QT App
        QtGui.QApplication.instance().exec_()

    def train_wrapped(self):

        self.logger.debug('Network Timestep: %d\n#Nodes: %d\n#Edges: %d' % (self.timestep, self.node_count, self.edge_count))

        # call overridden train function
        delta_w, x, n, s = self.train()

        # plot network, if viz is enabled
        if self.viz:
            self.draw()

        # increment timestep
        self.timestep += 1

        return delta_w, x, n, s

    def prepare(self):
        raise NotImplementedError

    def adapt(self, node, stimulus):
        raise NotImplementedError

    def edge_update(self, n, s):
        raise NotImplementedError

    def node_update(self, n, s, stimulus):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError