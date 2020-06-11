import random
import math
import numpy as np
import tensorflow as tf

class PTE(object):
    def __init__(self, graph, graph1, graph2, rep_size=128, batch_size=128, negative_ratio=5, learning_rate=0.001):
        self.cur_epoch = 0
        self.g = graph
        self.g1 = graph1
        self.g2 = graph2
        self.node_size = graph.G.number_of_nodes()
        self.node_size1 = graph1.G.number_of_nodes()
        self.node_size2 = graph2.G.number_of_nodes()
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.lr = learning_rate
        self.subgraph_1_list, self.subgraph_2_list = self._look_up_subgraph_vertex_in_graph(self.g, self.g1, self.g2)

        self.sampling_table_1, self.edge_alias_1, self.edge_prob_1 = self._gen_sampling_table(self.g1)
        self.sampling_table_2, self.edge_alias_2, self.edge_prob_2 = self._gen_sampling_table(self.g2)

        self.h, self.t, self.sign, self.embeddings, self.context_embeddings, self.h_e, self.t_e, self.t_e_context = self._create_variables()
      

        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.sign * tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1)))
        

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def _look_up_subgraph_vertex_in_graph(self, graph, subgraph_1, subgraph_2):
        sub_1, sub_2 = [], []
        graph_look_back_list = graph.look_back_list
        subgraph_1_look_back_list = subgraph_1.look_back_list
        subgraph_2_look_back_list = subgraph_2.look_back_list
        graph_dict = {}
        for index, item in enumerate(graph_look_back_list):
            graph_dict[item] = index
        for item in subgraph_1_look_back_list:
            sub_1.append(graph_dict[item])
        for item in subgraph_2_look_back_list:
            sub_2.append(graph_dict[item])
        return sub_1, sub_2

    def _gen_sampling_table(self, g):
        table_size = 1e8
        power = 0.75
        numNodes = g.node_size
        # negative sampling
        node_degree = np.zeros(numNodes)
        look_up = g.look_up_dict
        for edge in g.G.edges():
            node_degree[look_up[edge[0]]] += g.G[edge[0]][edge[1]]["weight"] 
        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])
        sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes):
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                sampling_table[i] = j
                i += 1

        data_size = g.G.number_of_edges()
        edge_alias = np.zeros(data_size, dtype=np.int32)
        edge_prob = np.zeros(data_size, dtype=np.float32)
        large_block = np.zeros(data_size, dtype=np.int32)
        small_block = np.zeros(data_size, dtype=np.int32)

        total_sum = sum([g.G[edge[0]][edge[1]]["weight"]
                         for edge in g.G.edges()])
        norm_prob = [g.G[edge[0]][edge[1]]["weight"] *
                     data_size / total_sum for edge in g.G.edges()]
        num_small_block = 0
        num_large_block = 0
        cur_small_block = 0
        cur_large_block = 0
        for k in range(data_size - 1, -1, -1):
            if norm_prob[k] < 1:
                small_block[num_small_block] = k
                num_small_block += 1
            else:
                large_block[num_large_block] = k
                num_large_block += 1
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block[num_small_block]
            num_large_block -= 1
            cur_large_block = large_block[num_large_block]
            edge_prob[cur_small_block] = norm_prob[cur_small_block]
            edge_alias[cur_small_block] = cur_large_block
            norm_prob[cur_large_block] = norm_prob[cur_large_block] + \
                                         norm_prob[cur_small_block] - 1
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block += 1

        while num_large_block:
            num_large_block -= 1
            edge_prob[large_block[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            edge_prob[small_block[num_small_block]] = 1

        return sampling_table, edge_alias, edge_prob

    def _create_variables(self):
        with tf.name_scope("variables"):
            self.h = tf.placeholder(tf.int32, [None])
            self.t = tf.placeholder(tf.int32, [None])
            self.sign = tf.placeholder(tf.float32, [None])
            self.embeddings = tf.Variable(tf.truncated_normal(shape=[self.node_size, self.rep_size], mean=0.0, stddev=0.01), name='vertex_embeddings',dtype=tf.float32, trainable=True)
            self.context_embeddings =  tf.Variable(tf.truncated_normal(shape=[self.node_size, self.rep_size], mean=0.0, stddev=0.01), name='vertex_embeddings', dtype=tf.float32, trainable=True)

            self.h_e = tf.nn.embedding_lookup(self.embeddings, self.h)
            self.t_e = tf.nn.embedding_lookup(self.embeddings, self.t)
            self.t_e_context = tf.nn.embedding_lookup(self.context_embeddings, self.t)

            return self.h, self.t, self.sign, self.embeddings, self.context_embeddings, self.h_e, self.t_e, self.t_e_context

def train_one_epoch(sess, model):
    sum_loss = 0.0
    num_batches1, num_batches2 = 0.0, 0.0
    batches1 = batch_iter("g1", model)
    for batch in batches1:
        num_batches1 += 1
        h, t, sign = batch
        feed_dict = {
            model.h: h,
            model.t: t,
            model.sign: sign
        }
        _, cur_loss = sess.run([model.train_op, model.loss], feed_dict)
        sum_loss += cur_loss

    batches2 = batch_iter("g2", model)
    for batch in batches2:
        num_batches2 += 1
        h, t, sigh = batch
        feed_dict = {
            model.h: h,
            model.t: t,
            model.sign: sigh}
        _, cur_loss = sess.run([model.train_op, model.loss], feed_dict)
        sum_loss += cur_loss

    avg_loss = sum_loss / (num_batches1 + num_batches2)
    print('epoch:{} avg of loss:{!s}'.format(model.cur_epoch, avg_loss))
    model.cur_epoch += 1

def batch_iter(scope, model):
    if scope == "g1":
        g = model.g1
        edge_prob = model.edge_prob_1
        edge_alias = model.edge_alias_1
        sampling_table = model.sampling_table_1
    if scope == "g2":
        g = model.g2
        edge_prob = model.edge_prob_2
        edge_alias = model.edge_alias_2
        sampling_table = model.sampling_table_2

    look_up = g.look_up_dict

    table_size = 1e8

    edges = [(look_up[x[0]], look_up[x[1]]) for x in g.G.edges()]
    data_size = g.G.number_of_edges()
    shuffle_indices = np.random.permutation(np.arange(data_size))

    # positive or negative mod
    mod = 0
    mod_size = 1 + model.negative_ratio
    h = []

    start_index = 0
    end_index = min(start_index + model.batch_size, data_size)
    while start_index < data_size:
        if mod == 0:
            sign = 1.
            h = []
            t = []
            for i in range(start_index, end_index):  # alias method solve the problem of random type probability
                if not random.random() < edge_prob[shuffle_indices[i]]:
                    shuffle_indices[i] = edge_alias[shuffle_indices[i]]
                cur_h = edges[shuffle_indices[i]][0]
                cur_t = edges[shuffle_indices[i]][1]
                if scope =="g1":
                    h.append(model.subgraph_1_list[cur_h])
                    t.append(model.subgraph_1_list[cur_t])

                if scope =="g2":
                    h.append(model.subgraph_2_list[cur_h])
                    t.append(model.subgraph_2_list[cur_t])

        else:
            sign = -1.
            t = []
            for i in range(len(h)):
                if scope =="g1":
                    t.append(model.subgraph_1_list[sampling_table[random.randint(0, table_size - 1)]])
                   
                if scope =="g2":
                    t.append(model.subgraph_2_list[sampling_table[random.randint(0, table_size - 1)]])
                   
        yield h, t, [sign]
        mod += 1
        mod %= mod_size
        if mod == 0:
            start_index = end_index
            end_index = min(start_index + model.batch_size, data_size)



def get_embeddings(model, g, sess):
    vectors ={}
    embeddings = model.embeddings.eval(session=sess)
    look_back = g.look_back_list
    for i, embedding in enumerate(embeddings):
        vectors[look_back[i]] = embedding

    return vectors

def save_embeddings(model, vectors, filename):
    fout = open(filename, 'w')
    node_num = len(vectors.keys())
    fout.write("{} {}\n".format(node_num, model.rep_size))
    for node, vec in vectors.items():
        fout.write("{} {}\n".format(node,
                                    ' '.join([str(x) for x in vec])))
    fout.close()