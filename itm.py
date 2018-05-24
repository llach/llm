from topological_map import TopologicalMap

import numpy as np

from util.example_data import circle

class ITM(TopologicalMap):

    def __init__(self, *args, **kwargs):
        self.name = "ITM - Instantaneous Topological Map"
        super(ITM, self).__init__(*args, **kwargs)
        self.eta = kwargs.get('eta', 0.1)
        self.r_max = kwargs.get('r_max', None)
        self.node_min = 2


    def prepare(self):
        # determine average distance between two consecutive data points
        if self.r_max is None:
            dist = 0
            for x in zip(data[:-1],data[1:]):
                dist += self.dist(x[0], x[1])
            self.r_max = 0.5*(dist/len(data))
            self.logger.debug("r_max: %f" %self.r_max)
        self.add_node(self.data[0])
        self.add_node(self.data[1])
        self.timestep += 2

    def adapt(self, node, stimulus):
        node.pos += self.eta * np.subtract(stimulus, node.pos)

    def edge_update(self, n, s):
        self.add_edge(n, s)
        dead_edges = set([])
        for c in n.neighbors:
            if np.dot(np.subtract(n.pos, s.pos), np.subtract(c.pos, s.pos)) < 0:
                dead_edges.add(n.get_edge_to_neighbor(c))

        for d in dead_edges:
            self.remove_edge(d)


        dead_nodes = set([])

        for node in self.nodes:
            if len(node.neighbors) == 0:
                dead_nodes.add(node)
        for node in dead_nodes:
            self.remove_node(node)

    def node_update(self, n, s, stimulus):

        thales = np.dot(np.subtract(n.pos, stimulus), np.subtract(s.pos, stimulus))
        dns = self.dist(stimulus, n.pos)
        if  thales > 0 and dns > self.r_max:
            new_node = self.add_node(position=stimulus)
            self.add_edge(new_node, n)

        if self.dist(n.pos, s.pos) < 0.5*self.r_max:
            self.remove_node(s)


    def train(self):
        if self.timestep > len(self.data):
            return

        x = self.data[self.timestep-1]
        # matching
        nearest, second = self.find_nearest(x)

        # adapt winner
        self.adapt(nearest, x)

        # update edges
        self.edge_update(nearest, second)

        # update nodes
        self.node_update(nearest, second, x)

if __name__=='__main__':
    itm = ITM(eta=0.1,  loggerlevel='DEBUG')
    data, labels = circle(samples_per_class=666, numclasses=3, shift_rad=2)
    data = np.append(data, labels, axis=1)
    itm.run(data)