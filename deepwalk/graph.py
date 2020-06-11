import networkx as nx
import numpy as np
import scipy.sparse as np

class Graph(object):
    def __init__(self):
        self.G = None
        self.look_up_dict = {}
        self.node_size = 0
        self.look_back_list = []

    def encode_node(self):
        for node in self.G.nodes():
            self.look_up_dict[node] = self.node_size
            self.look_back_list.append(node)
            self.node_size += 1


    def read_adjlist(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
        the format of each line: v1 n1 n2 n3 ... nk
        :param filename: the filename of input file
        """
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0

        self.encode_node()

    def read_edgelist(self, filename, weighted=False, directed=False):
        self.G = nx.DiGraph()

        if directed:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)
        fin = open(filename, 'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
        fin.close()
        self.encode_node()