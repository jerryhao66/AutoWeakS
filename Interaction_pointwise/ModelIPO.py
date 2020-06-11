from Interaction_pointwise import gendata
from Interaction_pointwise import setting, pretrain_vector
import logging
import tensorflow as tf
import os
import numpy as np
from time import time
import math
import heapq

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def print_logging(message):
    logging.info(message)
    print(message)


class IPointModelNetwork(object):
    def __init__(self):
        self.n_bins = 11
        self.epsilon = 0.00001
        self.lamb = 0.5
        self._prepare_hyperpara()
        self._prepare_data()
        self._build_graph()

    def _kernal_mus(self, n_kernels, use_exact):
        if use_exact:
            l_mu = [1]
        else:
            l_mu = [2]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)
        l_mu.append(1 - bin_size / 2)
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def _kernel_sigmas(self, n_kernels, lamb, use_exact):
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.00001]
        if n_kernels == 1:
            return l_sigma

        l_sigma += [bin_size * lamb] * (n_kernels - 1)
        return l_sigma

    def _weight_variable(self, shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random_uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial)

    def _prepare_hyperpara(self):
        self.weight_size = setting.WEIGHT_SIZE
        self.learning_rate = setting.LEARNING_RATE
        self.embedding_size = setting.EMB_DIM
        self.lambda_bilinear = setting.LAMBDA_BILINEAR
        self.mus = self._kernal_mus(self.n_bins, use_exact=True)
        self.sigmas = self._kernel_sigmas(self.n_bins, self.lamb, use_exact=True)

    def _prepare_data(self):
        self.vec_job, self.vec_course, self.vec_keyword, self.job_num, self.course_num, self.keywords_num = pretrain_vector.load_data(
            Deepwalk_path=setting.DEEPWALK_VEC_PATH, Line_path=setting.LINE_VEC_PATH,
            Metapath2vec_path=setting.METAPATH2VEC_PATH, Node2vec_path=setting.NODE2VEC_PATH,
            Pte_path=setting.PTE_VEC_PATH)
        # print_logging("Loaded keyword embeddings, #Keyword = %d" % self.keywords_num)

    def _create_placeholders(self):
        with tf.name_scope("IPoint_placeholder"):
            self.job_q = tf.placeholder(tf.int32, shape=[None, None], name='train_jobs_q')
            self.job_q_weights = tf.placeholder(tf.float32, shape=[None, None], name='jobs_idf')
            self.course_d = tf.placeholder(tf.int32, shape=[None, None], name='course_document')
            self.labels = tf.placeholder(tf.int32, shape=[None, 1], name='label')

    def _create_variables(self):
        with tf.name_scope("IPoint_variables"):
            tmp = np.sqrt(6.0) / np.sqrt(self.keywords_num + self.embedding_size)
            self.c1 = tf.Variable(
                tf.random_uniform(shape=[self.keywords_num, self.embedding_size], minval=-tmp, maxval=tmp), name='c1',
                dtype=tf.float32, trainable=True)
            # self.c1 = tf.Variable(
            #     tf.constant(self.vec_keyword, dtype='float32', shape=[self.keywords_num, self.embedding_size]),
            #     name='c1', trainable=True)
            self.c2 = tf.constant(0.0, tf.float32, shape=[1, self.embedding_size], name='c2')
            self.keyword_embeddings = tf.concat([self.c1, self.c2], 0, name='keyword_embeddings')

            self.W1 = self._weight_variable([self.n_bins, self.weight_size])
            self.b1 = tf.Variable(tf.zeros([self.weight_size]), name='b1')

            self.W2 = self._weight_variable([self.weight_size, 1])
            self.b2 = tf.Variable(tf.zeros(1), name='b2')

    def _create_loss(self):
        with tf.name_scope("IPoint_loss"):
            _, self.output = self._model(self.job_q, self.course_d, self.job_q_weights, self.keyword_embeddings)
            self.loss = tf.losses.log_loss(self.labels, self.output) + self.lambda_bilinear * tf.reduce_sum(
                tf.square(self.keyword_embeddings))

    def _create_optimizer(self):
        with tf.name_scope("IPoint_optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=1e-8).minimize(self.loss)

    def _create_reward(self):
        with tf.name_scope("IPoint_reward"):
            self.one_minus_output = 1.0 - self.output
            self.reward_output_concat = tf.concat([self.one_minus_output, self.output], 1)
            self.classes = tf.constant(2)
            self.eval_labels = tf.placeholder(tf.int32, shape=[None, 1], name='eval_labels')
            self.labels_reduce_dim = tf.reduce_sum(self.eval_labels, 1)
            self.one_hot = tf.one_hot(self.labels_reduce_dim, self.classes)
            self.reward = tf.log(tf.reduce_sum((self.reward_output_concat * self.one_hot + 1e-15), 1))

    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        self._create_reward()
        # print_logging("already bulid the IPO graph...")

    def _model(self, inputs_q, inputs_d, q_weights, embeddings):
        '''
        param inputs_q: input queries i.e. job. [nbatch, qlen, emb_dim]
        param inputs_d: input documents i.e. course [nbatch, dlen, emb_dim]
        param d_mask: document mask [nbatch, qlen, emb_dim]
        param q_mask: query mask
        param mu: kernel mu values.
        param sigma: kernel sigma values.
        return: return the predicted score for each <query, document> in the batch
        '''

        q_embed = tf.nn.embedding_lookup(embeddings, inputs_q, name='qemb')
        d_embed = tf.nn.embedding_lookup(embeddings, inputs_d, name='demb')

        ## Uingram Model
        # normalize and compute similarity matrix
        normalized_q_embed = tf.nn.l2_normalize(q_embed, axis=2)
        normalized_d_embed = tf.nn.l2_normalize(d_embed, axis=2)
        tmp = tf.transpose(normalized_d_embed, perm=[0, 2, 1])

        # similarity matrix [n_batch, qlen, dlen]
        sim = tf.einsum('ijk,ikl->ijl', normalized_q_embed, tmp, name='similarity_matrix')
        sim_shape = tf.shape(sim)

        # compute gaussian kernel
        rs_sim = tf.reshape(sim, [sim_shape[0], sim_shape[1], sim_shape[2], 1])

        # compute Gaussian scores of each kernel
        mu = tf.reshape(self.mus, shape=[1, 1, self.n_bins])
        sigma = tf.reshape(self.sigmas, shape=[1, 1, self.n_bins])

        # compute Gaussian scores of each kernel
        tmp = tf.exp(-tf.square(tf.subtract(rs_sim, mu)) / (tf.multiply(tf.square(sigma), 2)))

        # sum up gaussian scores
        kde = tf.reduce_sum(tmp, [2])
        kde = tf.log(tf.maximum(kde, 1e-10)) * 0.005  # 0.01 used to scale down the data. (0.08，0.09)
        # [batch_size, qlen, n_bins]

        # aggregated query terms
        # q_weights = [1, 1, 0, 0...]. Works as a query word mask.
        # Support query-term weigting if set to continous values (e.g. IDF).
        q_weights_shape = tf.shape(q_weights)
        q_weights = tf.reshape(q_weights, [q_weights_shape[0], q_weights_shape[1], 1])
        aggregated_kde = tf.reduce_sum(kde * q_weights, [1])  # [batch, n_bins]

        layer1 = tf.matmul(aggregated_kde, self.W1) + self.b1
        o = tf.nn.sigmoid(tf.matmul(layer1, self.W2) + self.b2)

        return sim, o


def debug(batch_index, model, sess, train_data):
    for index in batch_index:
        job, course, job_tf_idf, label = gendata.batch_gen(train_data, index)
        feed_dict = {model.job_q: job, model.job_q_weights: job_tf_idf, model.course_d: course,
                     model.labels: np.array(label).reshape(-1, 1)}
        output = sess.run(model.output, feed_dict)
        print(output)


def training_batch(batch_index, model, sess, train_data):
    for index in batch_index:
        job, course, job_tf_idf, label = gendata.batch_gen(train_data, index)
        feed_dict = {model.job_q: job, model.job_q_weights: job_tf_idf, model.course_d: course,
                     model.labels: np.array(label).reshape(-1, 1)}
        sess.run([model.loss, model.optimizer], feed_dict)


def training_loss(model, sess, train_data):
    train_loss = 0.0
    num_batch = len(train_data[1])
    for index in range(num_batch):
        job, course, job_tf_idf, label = gendata.batch_gen(train_data, index)
        feed_dict = {model.job_q: job, model.job_q_weights: job_tf_idf, model.course_d: course,
                     model.labels: np.array(label).reshape(-1, 1)}
        train_loss += sess.run(model.loss, feed_dict)
    return train_loss / num_batch


def evaluate(test_batch_index, model, sess, test_data):
    hits5, ndcgs5, hits20, ndcgs20, hits10, ndcgs10, maps, mrrs = [], [], [], [], [], [], [], []
    test_dict = {}
    for index in test_batch_index:
        test_job, test_course, test_job_tf_idf = gendata.evaluate_batch_gen(test_data, index)
        feed_dict = {model.job_q: test_job, model.job_q_weights: test_job_tf_idf, model.course_d: test_course}
        per_batch_predict = sess.run([model.output], feed_dict)
        per_batch_predict = np.array(per_batch_predict)
        temp = list(np.reshape(per_batch_predict, (-1)))
        gtItem = 0
        map_course_score = {t: temp[t] for t in range(setting.TEST_BATCH_SIZE)}
        test_dict[index] = map_course_score 
        ranklist5 = heapq.nlargest(5, map_course_score, key=map_course_score.get)
        ranklist20 = heapq.nlargest(20, map_course_score, key=map_course_score.get)
        ranklist10 = heapq.nlargest(10, map_course_score, key=map_course_score.get)
        ranklist100 = heapq.nlargest(100, map_course_score, key=map_course_score.get)
        hr5 = getHitRatio(ranklist5, gtItem)
        ndcg5 = getNDCG(ranklist5, gtItem)
        hr20 = getHitRatio(ranklist20, gtItem)
        ndcg20 = getNDCG(ranklist20, gtItem)
        hr10 = getHitRatio(ranklist10, gtItem)
        ndcg10 = getNDCG(ranklist10, gtItem)
        ap = getAP(ranklist100, gtItem)
        mrr = getMRR(ranklist100, gtItem)
        hits5.append(hr5)
        ndcgs5.append(ndcg5)
        hits20.append(hr20)
        ndcgs20.append(ndcg20)
        hits10.append(hr10)
        ndcgs10.append(ndcg10)
        maps.append(ap)
        mrrs.append(mrr)
    final_hr5, final_ndcg5, final_hr20, final_ndcg20, final_hr10, final_ndcg10, final_map, final_mrr = np.array(
        hits5).mean(), np.array(ndcgs5).mean(), np.array(hits20).mean(), np.array(
        ndcgs20).mean(), np.array(hits10).mean(), np.array(ndcgs10).mean(), np.array(maps).mean(), np.array(mrrs).mean()
    return (final_hr5, final_ndcg5, final_hr20, final_ndcg20, final_hr10, final_ndcg10, final_map, final_mrr, test_dict)


def predict(test_batch_index, model, sess, test_data):
    batch_predict_list = []
    for index in test_batch_index:
        test_job, test_course, test_job_tf_idf = gendata.evaluate_batch_gen(test_data, index)
        feed_dict = {model.job_q: test_job, model.job_q_weights: test_job_tf_idf, model.course_d: test_course}
        per_batch_predict = sess.run([model.output], feed_dict)
        batch_predict_list.append(np.reshape(np.array(per_batch_predict[0]), (-1)))

    return batch_predict_list


def training(model, sess, saver):
    best_hr5, best_ndcg5, best_hr10, best_ndcg10, best_map, best_mrr = -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
    best_loss = 2.0
    generate_batch_time = time()
    train_data = gendata.generate_batch()
    test_data = gendata.generate_test_batch()
    batch_time = time() - generate_batch_time
    num_batch = len(train_data[1])
    test_num_batch = len(test_data[1])
    batch_index = range(num_batch)
    test_batch_index = range(test_num_batch)

    for epoch_count in range(setting.EPOCH):
        train_begin = time()
        training_batch(batch_index, model, sess, train_data)
        # debug(batch_index, model, sess, train_data)

        train_time = time() - train_begin
        if epoch_count % setting.VERBOSE == 0:
            loss_begin = time()
            train_loss = training_loss(model, sess, train_data)
            loss_time = time() - loss_begin

            # print('epoch %d training loss is %.4f, training time is %.4f' % (epoch_count, train_loss, train_time))

            hr_5, ndcg_5, hr_20, ndcg_20, hr_10, ndcg_10, f_map, f_mrr, test_dict = evaluate(test_batch_index, model, sess,
                                                                                  test_data)
            # print(
            #     'epoch %d, training_loss is %.4f, training time is %.4fs, training loss time is %.4fs, generate batch time is %.4fs, hr5 = %.4f, ndcg5 = %.4f, hr20 = %.4f, ndcg20 = %.4f, hr10 = %.4f, ndcg10 = %.4f, map = %.4f, mrr = %.4f' % (
            #         epoch_count, train_loss, train_time, loss_time, batch_time, hr_5, ndcg_5, hr_20, ndcg_20, hr_10,
            #         ndcg_10, f_map, f_mrr))
            # a = predict(test_batch_index, model, sess, test_data)
            # if train_loss < best_loss:
            #     best_loss = train_loss
            # if hr_5 > best_hr5 or hr_10 > best_hr10:
            #     best_hr5, best_ndcg5, best_hr10, best_ndcg10, best_map, best_mrr = hr_5, ndcg_5, hr_10, ndcg_10, f_map, f_mrr
            #     saver.save(sess, setting.RANK_MODEL_CHECKPOINT_PATH, global_step=epoch_count)
        re_genecate_batch_begin_time = time()
        train_data = gendata.generate_batch()
        batch_time = time() - re_genecate_batch_begin_time
    # return best_hr5, best_ndcg5, best_hr10, best_ndcg10, best_map, best_mrr
    return test_dict

def init_test_dict():
    test_num_batch = 1289
    test_dict = {}
    for index in range(test_num_batch):
        map_course_score = {t: 0.0 for t in range(setting.TEST_BATCH_SIZE)}
        test_dict[index] = map_course_score

    return test_dict

def cal_reward(model, sess, saver):
    batch_reward_likelihood = []
    eval_data = gendata.generate_eval_batch()
    eval_num_batch = len(eval_data[1])

    saver.restore(sess, tf.train.get_checkpoint_state(
        os.path.dirname(setting.RANK_MODEL_CHECKPOINT_PATH + 'checkpoint')).model_checkpoint_path)
    # print("calculate reward, already load the ipo model...")

    for index in range(eval_num_batch):
        eval_job, eval_course, eval_job_tf_idf, eval_label = gendata.eval_reward_batch_gen(eval_data, index)
        feed_dict = {model.job_q: eval_job, model.job_q_weights: eval_job_tf_idf, model.course_d: eval_course,
                     model.eval_labels: eval_label}
        per_batch_reward = sess.run([model.reward], feed_dict)
        per_batch_reward = np.array(per_batch_reward).reshape(-1)
        batch_reward_likelihood.append(per_batch_reward)
    batch_reward_likelihood = np.array(batch_reward_likelihood)
    # print(np.mean(batch_reward_likelihood))
    return np.mean(batch_reward_likelihood)


def cal_mrr(model, sess, saver):
    eval_data = gendata.generate_eval_batch()
    eval_num_batch = len(eval_data[1])
    saver.restore(sess, tf.train.get_checkpoint_state(
        os.path.dirname(setting.RANK_MODEL_CHECKPOINT_PATH + 'checkpoint')).model_checkpoint_path)
    # print("calculate mrr, already load the ipo model...")
    mrrs = []
    for index in range(eval_num_batch):
        eval_job, eval_course, eval_job_tf_idf, eval_label = gendata.eval_reward_batch_gen(eval_data, index)
        feed_dict = {model.job_q: eval_job, model.job_q_weights: eval_job_tf_idf, model.course_d: eval_course,
                     model.eval_labels: eval_label}
        per_batch_predict = sess.run([model.output], feed_dict)
        per_batch_predict = np.array(per_batch_predict).reshape(-1)
        temp = list(np.reshape(per_batch_predict, (-1)))
        gtItem = 0
        map_course_score = {t: temp[t] for t in range(setting.EVAL_BATCH_SIZE)}
        ranklist100 = heapq.nlargest(100, map_course_score, key=map_course_score.get)
        mrr = getMRR(ranklist100, gtItem)
        mrrs.append(mrr)
    final_mrr = math.log(np.array(mrrs).mean())
    return final_mrr

def cal_mrr_dict(model, sess, saver):
    eval_data = gendata.generate_eval_batch()
    eval_num_batch = len(eval_data[1])
    # print(eval_num_batch)
    saver.restore(sess, tf.train.get_checkpoint_state(
        os.path.dirname(setting.RANK_MODEL_CHECKPOINT_PATH + 'checkpoint')).model_checkpoint_path)
    # print("calculate mrr, already load the ipo model...")

    job_course_dict = {}

    for index in range(eval_num_batch):
        eval_job, eval_course, eval_job_tf_idf, eval_label = gendata.eval_reward_batch_gen(eval_data, index)
        feed_dict = {model.job_q: eval_job, model.job_q_weights: eval_job_tf_idf, model.course_d: eval_course,
                     model.eval_labels: eval_label}
        per_batch_predict = sess.run([model.output], feed_dict)
        per_batch_predict = np.array(per_batch_predict).reshape(-1)
        temp = list(np.reshape(per_batch_predict, (-1)))
        gtItem = 0
        map_course_score = {t: temp[t] for t in range(setting.EVAL_BATCH_SIZE)}

        job_course_dict[index] = map_course_score

    return job_course_dict


def init_mrr_dict():
    eval_data = gendata.generate_eval_batch()
    eval_num_batch = len(eval_data[1])

    job_course_dict = {}

    for index in range(eval_num_batch):
        map_course_score = {t: 0.0 for t in range(setting.EVAL_BATCH_SIZE)}

        job_course_dict[index] = map_course_score

    return job_course_dict


def load_job_list(filename):
    job_list = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            line = line.strip()
            arr = line.split('\t')
            job_list.append(arr[0])
            line = f.readline()
    return job_list


def load_course_list(filename):
    course_list = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            per_instance_list = []
            line = line.strip()
            arr = line.split('\t')
            course_list.append(arr[0])
            line = f.readline()
    return course_list


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0


def getAP(ranklist, gtItem):
    hits = 0
    sum_precs = 0
    for n in range(len(ranklist)):
        if ranklist[n] == gtItem:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / 1
    else:
        return 0


def getMRR(ranklist, gtItem):
    for index, item in enumerate(ranklist):
        if item == gtItem:
            return 1.0 / (index + 1.0)
    return 0
