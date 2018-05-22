# -*- coding: utf-8 -*-

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import uuid
import sys
import math
import random

# calculate euclidean distance for vectors x and y
def euclidean_dist(x, y):

    # check length
    if(len(x) is not len(y)):
        raise Exception('Vector length must align!')

    dist = 0

    # calculate euclidean distance
    for xi, yi in zip(x, y):
        dist += (xi - yi)**2

    dist = math.sqrt(dist)

    return dist 

class Node:

    def __init__(self, pos):

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

    # edge to n is returned or exception thrown
    def get_edge_to_neighbor(self, n):

        for e in self.edges:
            if e.has_node(n):
                return e

        raise Exception('neighbor %s is unknown!' % n.uuid)

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

    # checks wheter this edge connects given node n
    def has_node(self, n):

        if self.n0 is n or self.n1 is n:
            return True
        else:
            return False

    # update line after nodes position change
    def update_plot_item(self):
        self.line.setData(pos=np.array([self.n0.pos, self.n1.pos]))


class GNG:

    def __init__(self, eta_n=0.7, eta_c=0.4, age_max=5, alpha=.15, beta=.1,
                 node_intervall=2, dist=euclidean_dist, freq=5, debug=False):

        self.name = 'GNG - Growing Neural Gas'

        # set and check network parameter
        if eta_n <= eta_c:
            raise Exception('eta_n has to be larger than eta_c!')
        
        self.eta_n = eta_n
        self.eta_c = eta_c

        # network parameter
        self.age_max = age_max
        self.alpha = alpha
        self.beta = beta
        self.node_intervall = node_intervall

        # init node and edge sets
        self.nodes = set([])
        self.edges = set([])

        # store node with highest error to avoid costly searching
        self.q_node = None

        # save iteration we are in
        self.timestep = 0

        # store number of nodes and edges present in network
        self.node_count = 0
        self.edge_count = 0

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

        # update all edge positions
        for e in self.edges:
            e.update_plot_item()
        
    # simple node adding function
    def add_node(self, position):
        
        # create and add node to network
        n = Node(position)
        self.nodes.add(n)
        
        # increase node count
        self.node_count += 1

        if self.debug:
            print('created node %s' % n.uuid)

        return n

    def add_edge(self, n0, n1):

        # create edge
        e = Edge(n0, n1)
        self.edges.add(e)

        self.edge_count += 1

        if self.debug:
            print('created edge %s' % e.uuid)

        # add edge to plot
        self.w.addItem(e.line)

        return e

    def remove_node(self, node):
        
        # sanity check wheter node can be removed
        if len(node.neighbors) > 0 or len(node.edges) > 0:
            raise Exception('this node, %s, cannot be removed!' % node.uuid)

        self.node_count -= 1

        if self.debug:
            print('removing node %s' % node.uuid)

        self.nodes.remove(node)

    def remove_edge(self, e):

        if self.debug:
            print('removing edge %s' % e.uuid)

        # get edge ready for safe removal
        e.prepare_removal()

        self.edge_count -= 1

        # remove edge from set
        self.edges.remove(e)

        # remove edge from plot widget
        self.w.removeItem(e.line)

    def adapt(self, node, stimulus):
        
        # adapt winner position
        node.pos += self.eta_n * (np.subtract(stimulus, node.pos))

        # adapt neighbor positions
        for n in node.neighbors:
            n.pos += self.eta_c * (np.subtract(stimulus, n.pos))

    def edge_update(self, n, s):
         
        # reset or create edge n <--> s
        if s in n.neighbors:
            e = n.get_edge_to_neighbor(s)
            e.age = 0
        else:
            self.add_edge(n, s)

        dead_edges = set([])

        # increase and check edge ages
        for e in n.edges:
            e.age += 1
            
            if e.age > self.age_max:
                dead_edges.add(e)

        for de in dead_edges:
            self.remove_edge(de)

        dead_nodes = set([])

        # check for edgeless nodes. can be done more efficiently,
        # if only the winners neighbors are checked that had edges removed
        for node in self.nodes:
            if len(node.neighbors) == 0:
                dead_nodes.add(node)

        for dn in dead_nodes:
            self.remove_node(dn)


    def node_update(self, n, stimulus):

        # increase winners error
        n.error += self.dist(n.pos, stimulus)**2

        # update node with highest error 
        if self.q_node is None or n.error >= self.q_node.error:
            self.q_node = n

        # insert new node every X steps
        if self.timestep % self.node_intervall == 0:
            
            # first, find neighbor p with largest error 
            p_node = None

            for node in self.q_node.neighbors:
                if p_node is None or p_node.error > node.error:
                    p_node = node
            
            # place new node r between p and q
            r_node_position = .5*np.add(p_node.pos, self.q_node.pos)
            
            # create error reducing node r
            r_node = self.add_node(r_node_position)

            # replace edge q <---> p with edges q <-> r <-> p
            self.remove_edge(self.q_node.get_edge_to_neighbor(p_node))
            
            self.add_edge(self.q_node, r_node)
            self.add_edge(p_node, r_node)

            # adjust node errors
            self.q_node.error -= self.alpha * self.q_node.error
            p_node.error -= self.alpha * p_node.error
            r_node.error = .5 * (self.q_node.error + p_node.error)


        # finally, update all node errors
        for node in self.nodes:
            node.error -= self.beta * node.error
            
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

            # node is further away from stimulus then second
            if node_dist >= self.dist(second.pos, stimulus):
                continue
            # stimulus between nearest and second
            elif node_dist >= self.dist(nearest.pos, stimulus):
                second = node
                continue
            # stimulus nearer than nearest
            else:
                second = nearest
                nearest = node
            

        if self.debug:
            print('nearest: %s; second: %s' % (nearest.uuid, second.uuid))

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

        if self.debug:
            print('Network Timestep: %d\n#Nodes: %d\n#Edges: %d' % (self.timestep, self.node_count, self.edge_count))

        # randomly select datapoint
        stimulus = random.choice(self.data)

        # find n, s
        n, s = self.find_nearest(stimulus)

        # adapt winner
        self.adapt(n, stimulus)

        # update edges
        self.edge_update(n, s)

        # update nodes
        self.node_update(n, stimulus)

        # plot network
        self.draw()
        
        # increment timestep
        self.timestep += 1

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        gng = GNG(debug=False)

        data = np.random.rand(200, 3)

        gng.run(data)
        QtGui.QApplication.instance().exec_()


