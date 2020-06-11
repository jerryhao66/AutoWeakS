from Representation_pointwise import gendata
from Representation_pointwise import setting, pretrain_vector
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


class RPointModelNetwork(object):
    def __init__(self):
        self.epsilon = 0.00001
        self._prepare_hypepara()
        self._prepare_data()
        self._build_graph()

    def _prepare_hypepara(self):
        self.layer1_size = setting.LAYER1_SIZE
        self.learning_rate = setting.LEARNING_RATE
        self.embedding_size = setting.EMB_DIM
        self.lambda_bilinear = setting.LAMBDA_BILINEAR

    def _prepare_data(self):
        self.vec_job, self.vec_course, self.vec_keyword, self.job_num, self.course_num, self.keywords_num = pretrain_vector.load_data(
            Deepwalk_path=setting.DEEPWALK_VEC_PATH, Line_path=setting.LINE_VEC_PATH,
            Metapath2vec_path=setting.METAPATH2VEC_PATH, Node2vec_path=setting.NODE2VEC_PATH,
            Pte_path=setting.PTE_VEC_PATH)
        # self.keyword_emb, self.keywords_num= data_utils.load_data(Deepwalk_path=setting.DEEPWALK_VEC_PATH, Line_path=setting.LINE_VEC_PATH, Metapath2vec_path=setting.METAPATH2VEC_PATH, Node2vec_path=setting.NODE2VEC_PATH, Pte_path=setting.PTE_VEC_PATH)
        # print_logging("Loaded keyword embeddings, #Keyword = %d" % self.keywords_num)

    def _create_placeholders(self):
        with tf.name_scope("RPoint_placeholder"):
            self.job_q = tf.placeholder(tf.int32, shape=[None, None], name='train_jobs_q')
            self.job_q_weights = tf.placeholder(tf.float32, shape=[None, None], name='jobs_idf')
            self.course_d = tf.placeholder(tf.int32, shape=[None, None], name='pos_course_document')
            self.labels = tf.placeholder(tf.int32, shape=[None, 1], name='labels')

    def _create_varibales(self):
        with tf.name_scope("RPoint_variables"):
            tmp = np.sqrt(6.0) / np.sqrt(self.keywords_num + self.embedding_size)
            self.c1 = tf.Variable(
                tf.random_uniform(shape=[self.keywords_num, self.embedding_size], minval=-tmp, maxval=tmp), name='c1',
                dtype=tf.float32, trainable=True)
            # self.c1 = tf.Variable(
            #     tf.constant(self.vec_keyword, dtype='float32', shape=[self.keywords_num, self.embedding_size]),
            #     name='c1',
            #     trainable=True)
            self.c2 = tf.constant(0.0, tf.float32, shape=[1, self.embedding_size], name='c2')
            self.keyword_embeddings = tf.concat([self.c1, self.c2], 0, name='keyword_embeddings')

            ################# share same weight #############################################
            self.layer1_range_q = np.sqrt(6.0 / ((setting.MAX_PADDING_Q + 1) * self.embedding_size + self.layer1_size))
            self.layer1_range_d = np.sqrt(6.0 / ((setting.MAX_PADDING_D + 1) * self.embedding_size + self.layer1_size))

            self.W1 = tf.Variable(
                tf.random_uniform([(setting.MAX_PADDING_Q + 1) * self.embedding_size, self.layer1_size],
                                  -self.layer1_range_q,
                                  self.layer1_range_q), name='W1')

            self.W2 = tf.Variable(
                tf.random_uniform([(setting.MAX_PADDING_D + 1) * self.embedding_size, self.layer1_size],
                                  -self.layer1_range_d,
                                  self.layer1_range_d), name='W2')

            self.b1 = tf.Variable(tf.random_uniform([1, self.layer1_size], -self.layer1_range_q, self.layer1_range_q),
                                  name='b1')

            self.b2 = tf.Variable(tf.random_uniform([1, self.layer1_size], -self.layer1_range_d, self.layer1_range_d),
                                  name='b2')

    def _ncf_create_inference(self, input_job, input_course, input_weights, keyword_embeddings):
        with tf.name_scope("RPoint_inference"):
            '''
            input_job b*q  input_course b*d q==d
            '''
            q_embed = tf.nn.embedding_lookup(keyword_embeddings, input_job, name='qemb')  # (b, q, e)
            d_embed = tf.nn.embedding_lookup(keyword_embeddings, input_course, name='demb')  # (b, d, e)

            input_weights_shape = tf.shape(input_weights)
            input_weights = tf.reshape(input_weights, [input_weights_shape[0], input_weights_shape[1], 1])  # (b, q, 1)

            # normalized_q_embed = tf.nn.l2_normalize(q_embed, axis=2)
            # normalized_d_embed = tf.nn.l2_normalize(d_embed, axis=2)
            q_embed = q_embed * input_weights

            b_q = tf.shape(q_embed)[0]
            b_d = tf.shape(d_embed)[0]
            q = tf.shape(q_embed)[1]
            d = tf.shape(d_embed)[1]

            layer1_q_output = tf.nn.relu(tf.matmul(tf.reshape(q_embed, [b_q, -1]),
                                                   self.W1) + self.b1)  # ((b,  q*e) *(q*e, l1) + (1, l1) = (b, l1)
            layer1_d_output = tf.nn.relu(tf.matmul(tf.reshape(d_embed, [b_d, -1]),
                                                   self.W2) + self.b2)  # ((b, d*e) *(d*e, l1) + (1, l1) = (b, l1)

            normalized_q_embed = tf.nn.l2_normalize(layer1_q_output, axis=1)  # (b, l2)  0-1
            normalized_d_embed = tf.nn.l2_normalize(layer1_d_output, axis=1)  # (b, l2)  0-1

            sim = tf.reduce_sum(normalized_q_embed * normalized_d_embed, axis=1, keepdims=True)  # (b, 1)

            return sim

    def _create_loss(self):
        with tf.name_scope("Rpoint_loss"):
            self.output = self._ncf_create_inference(self.job_q, self.course_d, self.job_q_weights,
                                                     self.keyword_embeddings)
            self.loss = tf.losses.log_loss(self.labels, self.output) + self.lambda_bilinear * tf.reduce_sum(
                tf.square(self.keyword_embeddings))

    def _create_reward(self):
        with tf.name_scope("RPoint_reward"):
            self.one_minus_output = 1.0 - self.output
            self.reward_output_concat = tf.concat([self.one_minus_output, self.output], 1)
            self.classes = tf.constant(2)
            self.eval_labels = tf.placeholder(tf.int32, shape=[None, 1], name='eval_labels')
            self.labels_reduce_dim = tf.reduce_sum(self.eval_labels, 1)
            self.one_hot = tf.one_hot(self.labels_reduce_dim, self.classes)

            self.reward = tf.log(tf.reduce_sum((self.reward_output_concat * self.one_hot + 1e-15), 1))

    def _create_optimizer(self):
        with tf.name_scope("Rpoint_optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    epsilon=self.epsilon).minimize(
                self.loss)

    def _build_graph(self):
        self._create_placeholders()
        self._create_varibales()
        self._create_loss()
        self._create_reward()
        self._create_optimizer()
        logging.info("already build the representation pointwise computing graph...")


def training_batch(batch_index, model, sess, train_data):
    for index in batch_index:
        job, course, job_tf_idf, labels = gendata.batch_gen(train_data, index)

        feed_dict = {model.job_q: job, model.job_q_weights: job_tf_idf, model.course_d: course,
                     model.labels: np.array(labels).reshape(-1, 1)}
        sess.run([model.loss, model.optimizer], feed_dict)


def training_loss(model, sess, train_data):
    train_loss = 0.0
    num_batch = len(train_data[1])
    for index in range(num_batch):
        job, course, job_tf_idf, labels = gendata.batch_gen(train_data, index)
        feed_dict = {model.job_q: job, model.job_q_weights: job_tf_idf, model.course_d: course,
                     model.labels: np.array(labels).reshape(-1, 1)}
        train_loss += sess.run(model.loss, feed_dict)
    return train_loss / num_batch


def predict(test_batch_index, model, sess, test_data):
    batch_predict_list = []
    for index in test_batch_index:
        test_job, test_course, test_job_tf_idf = gendata.evaluate_batch_gen(test_data, index)
        feed_dict = {model.job_q: test_job, model.job_q_weights: test_job_tf_idf, model.course_d: test_course}
        per_batch_predict = sess.run([model.output], feed_dict)
        batch_predict_list.append(np.reshape(np.array(per_batch_predict[0]), (-1)))

    return batch_predict_list


def evaluate(test_batch_index, model, sess, test_data):
    hits5, ndcgs5, hits20, ndcgs20, hits10, ndcgs10, maps, mrrs = [], [], [], [], [], [], [], []
    test_dict = {}
    for index in test_batch_index:
        test_job, test_course, test_job_tf_idf = gendata.evaluate_batch_gen(test_data, index)
        feed_dict = {model.job_q: test_job, model.job_q_weights: test_job_tf_idf, model.course_d: test_course}
        per_batch_predict = sess.run([model.output], feed_dict)
        per_batch_predict = np.array(per_batch_predict)

        temp = list(np.reshape(per_batch_predict, (-1)))  # (100,)

        # print(temp) #list[...]
        gtItem = 0
        map_course_score = {t: temp[t] for t in range(setting.TEST_BATCH_SIZE)}
        test_dict[index] = map_course_score
        # print(map_course_score)
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

# def evaluate_dict(test_batch_index, model, sess, test_data):
#     hits5, ndcgs5, hits20, ndcgs20, hits10, ndcgs10, maps, mrrs = [], [], [], [], [], [], [], []
#     test_dict = {}
#     for index in test_batch_index:
#         test_job, test_course, test_job_tf_idf = gendata.evaluate_batch_gen(test_data, index)
#         feed_dict = {model.job_q: test_job, model.job_q_weights: test_job_tf_idf, model.course_d: test_course}
#         per_batch_predict = sess.run([model.output], feed_dict)
#         per_batch_predict = np.array(per_batch_predict)

#         temp = list(np.reshape(per_batch_predict, (-1)))  # (100,)

#         # print(temp) #list[...]
#         gtItem = 0
#         map_course_score = {t: temp[t] for t in range(setting.TEST_BATCH_SIZE)}
#         test_dict[index] = map_course_score
#         # print(map_course_score)
       
#     return test_dict


def training(model, sess, saver):
    best_hr5, best_ndcg5, best_hr10, best_ndcg10, best_map, best_mrr = -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
    best_loss = 2.0
    generate_batch_time = time()
    train_data = gendata.generate_batch()
    test_data = gendata.generate_test_batch()
    batch_time = time() - generate_batch_time
    num_batch = len(train_data[1])
    test_num_batch = len(test_data[1])
    # print(test_num_batch)
    batch_index = range(num_batch)
    test_batch_index = range(test_num_batch)

    for epoch_count in range(setting.EPOCH):
        train_begin = time()
        training_batch(batch_index, model, sess, train_data)

        train_time = time() - train_begin
        if epoch_count % setting.VERBOSE == 0:
            loss_begin = time()
            train_loss = training_loss(model, sess, train_data)
            loss_time = time() - loss_begin
            # print('epoch %d training loss is %.4f, training time is %.4f' % (epoch_count, train_loss, train_time))
            eval_begin = time()
            # test_dict = evaluate_dict(test_batch_index, model, sess, test_data)
            hr_5, ndcg_5, hr_20, ndcg_20, hr_10, ndcg_10, f_map, f_mrr, test_dict = evaluate(test_batch_index, model, sess,
                                                                                  test_data)
            evaluate_time = time() - eval_begin
            # print(
            #     'epoch %d, training_loss is %.4f, training time is %.4fs, training loss time is %.4fs, generate batch time is %.4fs, evaluate time is %.4fs, hr5 = %.4f, ndcg5 = %.4f, hr20 = %.4f, ndcg20 = %.4f, hr10 = %.4f, ndcg10 = %.4f, map = %.4f, mrr = %.4f' % (
            #         epoch_count, train_loss, train_time, loss_time, batch_time, evaluate_time, hr_5, ndcg_5, hr_20,
            #         ndcg_20, hr_10, ndcg_10, f_map, f_mrr))
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
    # print("calculate reward, already load the rpo model...")

    for index in range(eval_num_batch):  # per job
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
    # print("calculate mrr, already load the rpo model...")
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
    # print("calculate mrr, already load the rpo model...")

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
