# -*- coding: utf-8 -*-

from topological_map import TopologicalMap

import numpy as np
import random

class GNG(TopologicalMap):

    def __init__(self, *args, **kwargs):

        self.name = 'GNG - Growing Neural Gas'

        self.node_min = kwargs.get('node_min', 2)
        self.node_max = kwargs.get('node_max', 100)

        # init from superclass
        super(GNG, self).__init__(*args, node_min=self.node_min, node_max=self.node_max, **kwargs)

        # network parameter
        self.eta_n = kwargs.get('eta_n', .7)
        self.eta_c = kwargs.get('eta_c', .4)

        self.alpha = kwargs.get('alpha', .15)
        self.beta = kwargs.get('beta', .1)

        self.age_max = kwargs.get('age_max', 5)
        self.node_intervall = kwargs.get('node_intervall', 2) # aka lambda

        # set and check network parameter
        if self.eta_n <= self.eta_c:
            raise Exception('eta_n has to be larger than eta_c!')

        # store node with highest error to avoid costly searching
        self.q_node = None

    def prepare(self):

        # add two initial nodes
        self.add_node(random.choice(self.data))
        self.add_node(random.choice(self.data))

        # increment timestep
        self.timestep += 2

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


    def node_update(self, n, s, stimulus):

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

            # be aware of node max!
            if r_node is None:
                return

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
    
    def train(self):

        # randomly select datapoint
        stimulus = random.choice(self.data)

        # find n, s
        n, s = self.find_nearest(stimulus)

        # adapt winner
        self.adapt(n, stimulus)

        # update edges
        self.edge_update(n, s)

        # update nodes
        self.node_update(n, s, stimulus)


if __name__ == '__main__':
    data = np.random.rand(200, 3)

    gng = GNG(debug=True)
    gng.run(data)



