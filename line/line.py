from __future__ import print_function
import random
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from linealgo.evaluate import evaluate



class _LINE(object):

    def __init__(self, graph, rep_size=128, batch_size=1000, negative_ratio=5, order=3, learning_rate=0.001):
        self.cur_epoch = 0
        self.order = order
        self.g = graph
        self.node_size = graph.G.number_of_nodes()
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.lr = learning_rate

        self.gen_sampling_table()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=False, seed=cur_seed)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            self.build_graph()
        

    def build_graph(self):
        self.h = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.sign = tf.placeholder(tf.float32, [None])

        cur_seed = random.getrandbits(32) # return a number between 0~2^32
        self.embeddings = tf.get_variable(name="embeddings"+str(self.order), shape=[
                                          self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        self.context_embeddings = tf.get_variable(name="context_embeddings"+str(self.order), shape=[
                                                  self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        self.h_e = tf.nn.embedding_lookup(self.embeddings, self.h)
        self.t_e = tf.nn.embedding_lookup(self.embeddings, self.t)
        self.t_e_context = tf.nn.embedding_lookup(
            self.context_embeddings, self.t)
        self.second_loss = -tf.reduce_mean(tf.log_sigmoid(
            self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1)))
        self.first_loss = -tf.reduce_mean(tf.log_sigmoid(
            self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)))
        if self.order == 1:
            self.loss = self.first_loss
        else:
            self.loss = self.second_loss
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def train_one_epoch(self, sess):
        sum_loss = 0.0
        batches = self.batch_iter()
        batch_id = 0
        for batch in batches:
            h, t, sign = batch
            feed_dict = {
                self.h: h,
                self.t: t,
                self.sign: sign,
            }
            _, cur_loss = sess.run([self.train_op, self.loss], feed_dict)
            sum_loss += cur_loss
            batch_id += 1
        avg_loss = sum_loss / batch_id
        print('epoch:{} avg of loss:{!s}'.format(self.cur_epoch, avg_loss))
        self.cur_epoch += 1

    def batch_iter(self):
        look_up = self.g.look_up_dict

        table_size = 1e8
        numNodes = self.node_size

        edges = [(look_up[x[0]], look_up[x[1]]) for x in self.g.G.edges()]

        data_size = self.g.G.number_of_edges()
        edge_set = set([x[0]*numNodes+x[1] for x in edges])
        shuffle_indices = np.random.permutation(np.arange(data_size))

        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0

        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)
        while start_index < data_size:
            if mod == 0:
                sign = 1.
                h = []
                t = []
                for i in range(start_index, end_index):   # alias method solve the problem of random type probability
                    if not random.random() < self.edge_prob[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
            else:
                sign = -1.
                t = []
                for i in range(len(h)):
                    t.append(
                        self.sampling_table[random.randint(0, table_size-1)])

            yield h, t, [sign]
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index+self.batch_size, data_size)

    def gen_sampling_table(self):
        table_size = 1e8
        power = 0.75
        numNodes = self.node_size

        #negative sampling
        node_degree = np.zeros(numNodes)  # out degree

        look_up = self.g.look_up_dict
        for edge in self.g.G.edges():
            node_degree[look_up[edge[0]]
                        ] += self.g.G[edge[0]][edge[1]]["weight"]

        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

        self.sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes):
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1

        data_size = self.g.G.number_of_edges()
        self.edge_alias = np.zeros(data_size, dtype=np.int32)
        self.edge_prob = np.zeros(data_size, dtype=np.float32)
        large_block = np.zeros(data_size, dtype=np.int32)
        small_block = np.zeros(data_size, dtype=np.int32)

        total_sum = sum([self.g.G[edge[0]][edge[1]]["weight"]
                         for edge in self.g.G.edges()])
        norm_prob = [self.g.G[edge[0]][edge[1]]["weight"] *
                     data_size/total_sum for edge in self.g.G.edges()]
        num_small_block = 0
        num_large_block = 0
        cur_small_block = 0
        cur_large_block = 0
        for k in range(data_size-1, -1, -1):
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
            self.edge_prob[cur_small_block] = norm_prob[cur_small_block]
            self.edge_alias[cur_small_block] = cur_large_block
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
            self.edge_prob[large_block[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            self.edge_prob[small_block[num_small_block]] = 1

    def get_embeddings(self, sess):
        vectors = {}
        embeddings = self.embeddings.eval(session=sess)
        look_back = self.g.look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        return vectors


class LINE(object):

    def __init__(self, args, sess, graph, rep_size, batch_size, epoch, negative_ratio, order):
        self.rep_size = rep_size
        self.order = order
        self.best_result = 0
        self.vectors = {}
        self.lr = args.lr
        self.graph = graph
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.epoch = epoch
        self.args = args
        self.sess = sess
        

    def training(self):
        best_hr10, best_ndcg10 = 0.0, 0.0
        if self.order == 3:
            self.model1 = _LINE(self.graph, self.rep_size/2, self.batch_size,
                                self.negative_ratio, order=1, learning_rate=self.lr)
            self.model2 = _LINE(self.graph, self.rep_size/2, self.batch_size,
                                self.negative_ratio, order=2, learning_rate=self.lr)

            LINE_saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                self.model1.train_one_epoch(self.sess)
                self.model2.train_one_epoch(self.sess)
                vectors_all = self.get_embeddings(self.sess)
                hr5, ndcg5, hr20, ndcg20, hr10, ndcg10, map_, mrr = evaluate(vectors_all, self.args)
                print('epoch %d, hr5 = %.4f, ndcg5 = %.4f, hr20 = %.4f, ndcg20 = %.4f, hr10 = %.4f, ndcg10 = %.4f, map = %.4f, mrr = %.4f' % (i, hr5, ndcg5, hr20, ndcg20, hr10, ndcg10, map_, mrr))
                if hr10 >= best_hr10 or ndcg10 >= best_ndcg10:
                    best_hr10 = hr10
                    best_ndcg10 = ndcg10
                    LINE_saver.save(self.sess, self.args.model_checkpoint_path, global_step=i)


        else:
            self.model = _LINE(self.graph, self.rep_size, self.batch_size,
                               self.negative_ratio, order=self.order, learning_rate=self.lr)
            LINE_saver = tf.train.Saver()
            for i in range(self.epoch):
                self.model.train_one_epoch(self.sess)
                vectors_all = self.get_embeddings(self.sess)
                hr5, ndcg5, hr20, ndcg20, hr10, ndcg10, map_, mrr = evaluate(vectors_all, self.args)
                print('epoch %d, hr5 = %.4f, ndcg5 = %.4f, hr20 = %.4f, ndcg20 = %.4f, hr10 = %.4f, ndcg10 = %.4f, map = %.4f, mrr = %.4f' % (i, hr5, ndcg5, hr20, ndcg20, hr10, ndcg10, map_, mrr))
                if hr10 >= best_hr10 or ndcg10 >= best_ndcg10:
                    best_hr10 = hr10
                    best_ndcg10 = ndcg10
                    LINE_saver.save(self.sess, self.args.model_checkpoint_path, global_step=i)   

        
    def get_embeddings(self, sess):
        self.last_vectors = self.vectors
        self.vectors = {}
        if self.order == 3:
            vectors1 = self.model1.get_embeddings(sess)
            vectors2 = self.model2.get_embeddings(sess)
            for node in vectors1.keys():
                self.vectors[node] = np.append(vectors1[node], vectors2[node])
        else:
            self.vectors = self.model.get_embeddings(sess)
        return self.vectors

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
