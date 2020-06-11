import tensorflow as tf
import os
import numpy as np
import heapq
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from bert_representation import setting

class BERT_Representation(object):
    def __init__(self):
        self.learning_rate = 0.01
        self.embedding_size = 768
        self.course_num = setting.COURSE_NUM
        self.job_num = setting.JOB_NUM
        self.weight_size = 400
        self.weight_size1 = 200
        self.lambda_bilinear = 1e-10
        self.epsilon = 1e-8
        self._build_graph()

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.job_input_id = tf.placeholder(tf.int32, shape=[None])
            self.course_input_id = tf.placeholder(tf.int32, shape=[None])
            self.labels = tf.placeholder(tf.int32, shape=[None, 1])

    def _weight_variable(self, shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random_uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial)

    def _create_variables(self):
        with tf.name_scope("embedding"):
            bert_pretrain_course_ebd = np.load(setting.BERT_PRETRAIN_COURSE_EBD)
            bert_pretrain_job_ebd = np.load(setting.BERT_PRETRAIN_JOB_EBD)
            self.course_embedding = tf.Variable(
                tf.constant(bert_pretrain_course_ebd, dtype='float32', shape=[self.course_num, self.embedding_size]),
                name='pre_train_bert_course_embedding', trainable=True)
            self.job_embedding = tf.Variable(
                tf.constant(bert_pretrain_job_ebd, dtype='float32', shape=[self.job_num, self.embedding_size]),
                name='pre_train_bert_job_embedding', trainable=True)

            self.W1 = self._weight_variable([self.embedding_size, self.weight_size])
            self.b1 = tf.Variable(tf.zeros([self.weight_size]), name='b1')

            self.W2 = self._weight_variable([self.embedding_size, self.weight_size])
            self.b2 = tf.Variable(tf.zeros(self.weight_size), name='b2')

            self.W3 = self._weight_variable([self.weight_size, self.weight_size1])
            self.b3 = tf.Variable(tf.zeros(self.weight_size1), name='b3')

            self.W4 = self._weight_variable([self.weight_size, self.weight_size1])
            self.b4 = tf.Variable(tf.zeros(self.weight_size1), name='b4')

    def _create_inference(self):
        with tf.name_scope("inference"):
            job_ebd = tf.nn.embedding_lookup(self.job_embedding, self.job_input_id)  # [b, e]
            course_ebd = tf.nn.embedding_lookup(self.course_embedding, self.course_input_id)  # [b, e]

            MLP_output_job = tf.nn.relu(tf.matmul(job_ebd, self.W1) + self.b1)  # [b, w1]
            MLP_output_course = tf.nn.relu(tf.matmul(course_ebd, self.W2) + self.b2)  # [b, w1]

            normalized_job_embed = tf.nn.l2_normalize(MLP_output_job, axis=1)  # (b, w1)  0-1
            normalized_course_embed = tf.nn.l2_normalize(MLP_output_course, axis=1)  # (b, w1)



            # MLP_output_job0 = tf.nn.relu(tf.matmul(job_ebd, self.W1) + self.b1)  
            # MLP_output_course0 = tf.nn.relu(tf.matmul(course_ebd, self.W2) + self.b2)  
            # MLP_output_job = tf.nn.relu(tf.matmul(MLP_output_job0, self.W3) + self.b3)
            # MLP_output_course = tf.nn.relu(tf.matmul(MLP_output_course0, self.W4) + self.b4)

            # normalized_job_embed = tf.nn.l2_normalize(MLP_output_job, axis=1)  
            # normalized_course_embed = tf.nn.l2_normalize(MLP_output_course, axis=1)  


            self.output = tf.reduce_sum(normalized_job_embed * normalized_course_embed, axis=1, keepdims=True)  # (b, 1)

    def _create_loss(self):
        with tf.name_scope("BERT_Pointwise_loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) + self.lambda_bilinear * tf.reduce_sum(
                tf.square(self.course_embedding)) + self.lambda_bilinear * tf.reduce_sum(
                tf.square(self.job_embedding))

    def _create_optimizer(self):
        with tf.name_scope("BERT_pointwise_optimizer"):
        	self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    epsilon=self.epsilon).minimize(self.loss)

    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()


def generate_train_data():
    job_list, course_list, label_list, job_course_dict = [], [], [], {}
    with open(setting.TRAIN_POSITIVE_PAIR_FILE, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            pos_job, pos_course = int(arr[0]), int(arr[1])
            if pos_job not in job_course_dict:
                job_course_dict[pos_job] = []
                job_course_dict[pos_job].append(pos_course)
            else:
                job_course_dict[pos_job].append(pos_course)

            line = f.readline()

    with open(setting.TRAIN_POSITIVE_PAIR_FILE, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            pos_job, pos_course = int(arr[0]), int(arr[1])

            # pos
            job_list.append(pos_job)
            course_list.append(pos_course)
            label_list.append(1)

            # neg
            for _ in range(setting.NEGATIVE_NUM):
                j = np.random.randint(setting.COURSE_NUM)
                while j in job_course_dict[pos_job]:
                    j = np.random.randint(setting.COURSE_NUM)

                job_list.append(pos_job)
                course_list.append(j)
                label_list.append(0)

            line = f.readline()
    return job_list, course_list, label_list


def generate_batch_data(job, course, label, index):
    batch_job = job[index * setting.TRAIN_BATCH_SIZE: index * setting.TRAIN_BATCH_SIZE + setting.TRAIN_BATCH_SIZE]
    batch_course = course[index * setting.TRAIN_BATCH_SIZE: index * setting.TRAIN_BATCH_SIZE + setting.TRAIN_BATCH_SIZE]
    batch_label = label[index * setting.TRAIN_BATCH_SIZE: index * setting.TRAIN_BATCH_SIZE + setting.TRAIN_BATCH_SIZE]
    return batch_job, batch_course, batch_label


def generate_valid_data():
    job_list, course_list = [], []
    with open(setting.VALID_POINTWISE_FILE, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            job, course = arr[0], arr[1]
            job_list.append(job)
            course_list.append(course)
            line = f.readline()
        return job_list, course_list


def generate_test_data():
    job_list, course_list = [], []
    with open(setting.TEST_POINTWISE_FILE, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            job, course = arr[0], arr[1]
            job_list.append(job)
            course_list.append(course)
            line = f.readline()
        return job_list, course_list


def generate_test_valid_batch_data(job, course, index):
    batch_job = job[index * setting.TEST_BATCH_SIZE: index * setting.TEST_BATCH_SIZE + setting.TEST_BATCH_SIZE]
    batch_course = course[index * setting.TEST_BATCH_SIZE: index * setting.TEST_BATCH_SIZE + setting.TEST_BATCH_SIZE]
    return batch_job, batch_course


def training(model, sess, saver):
    best_hr5, best_ndcg5, best_hr10, best_ndcg10, best_map, best_mrr, best_loss = -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.0
    train_job, train_course, train_label = generate_train_data()
    num_train_batch = int (len(train_job) / setting.TRAIN_BATCH_SIZE)

    valid_job, valid_course = generate_valid_data()
    num_valid_batch = int (len(valid_job) / setting.EVAL_BATCH_SIZE)

    test_job, test_course = generate_test_data()
    num_test_batch = int (len(test_job) / setting.TEST_BATCH_SIZE)

    train_batch_index = range(num_train_batch)
    valid_batch_index = range(num_valid_batch)
    test_batch_index = range(num_test_batch)

    for epoch_count in range(setting.EPOCH):
        training_batch(train_batch_index, model, sess, train_job, train_course, train_label)
        if epoch_count % setting.VERBOSE == 0:
            train_loss = training_loss(model, sess, train_job, train_course, train_label)
            hr_5, ndcg_5, hr_10, ndcg_10, f_map, f_mrr, test_dict = evaluate(test_batch_index, model, sess, test_job,
                                                                             test_course)
            # print(
            #     'epoch %d, training loss is %.4f, hr5 is %.4f, hr10 is %.4f, ndcg5 is %.4f, ndcg10 is %.4f, mrr is %.4f' % (
            #     epoch_count, train_loss, hr_5, hr_10, ndcg_5, ndcg_10, f_mrr))
            if hr_5 > best_hr5 or hr_10 > best_hr10:
                best_hr5, best_ndcg5, best_hr10, best_ndcg10, best_map, best_mrr = hr_5, ndcg_5, hr_10, ndcg_10, f_map, f_mrr
                saver.save(sess, setting.RANK_MODEL_CHECKPOINT_PATH, global_step=epoch_count)
        train_job, train_course, train_label = generate_train_data()
        np.random.permutation(train_batch_index)
    return test_dict


def training_batch(batch_index, model, sess, job, course, label):
    for index in batch_index:
        batch_job, batch_course, batch_label = generate_batch_data(job, course, label, index)
        feed_dict = {model.job_input_id: batch_job, model.course_input_id: batch_course,
                     model.labels: np.array(batch_label).reshape(-1, 1)}
        sess.run([model.loss, model.optimizer], feed_dict)


def training_loss(model, sess, job, course, label):
    train_loss = 0.0
    num_batch = int (len(job) / setting.TRAIN_BATCH_SIZE) 
    for index in range(num_batch):
        batch_job, batch_course, batch_label = generate_batch_data(job, course, label, index)
        feed_dict = {model.job_input_id: batch_job, model.course_input_id: batch_course,
                     model.labels: np.array(batch_label).reshape(-1, 1)}
        train_loss += sess.run(model.loss, feed_dict)
    return train_loss / num_batch


def evaluate(test_batch_index, model, sess, test_job, test_course):
    hits5, ndcgs5, hits10, ndcgs10, maps, mrrs = [], [], [], [], [], []
    test_dict = {}
    for index in test_batch_index:
        test_batch_job, test_batch_course = generate_test_valid_batch_data(test_job, test_course, index)
        feed_dict = {model.job_input_id: test_batch_job, model.course_input_id: test_batch_course}
        per_batch_predict = sess.run([model.output], feed_dict)
        per_batch_predict = np.array(per_batch_predict)
        temp = list(np.reshape(per_batch_predict, (-1)))
        gtItem = 0
        map_course_score = {t: temp[t] for t in range(setting.TEST_BATCH_SIZE)}
        test_dict[index] = map_course_score
        ranklist5 = heapq.nlargest(5, map_course_score, key=map_course_score.get)
        ranklist10 = heapq.nlargest(10, map_course_score, key=map_course_score.get)
        ranklist100 = heapq.nlargest(100, map_course_score, key=map_course_score.get)
        hr5 = getHitRatio(ranklist5, gtItem)
        ndcg5 = getNDCG(ranklist5, gtItem)
        hr10 = getHitRatio(ranklist10, gtItem)
        ndcg10 = getNDCG(ranklist10, gtItem)
        ap = getAP(ranklist100, gtItem)
        mrr = getMRR(ranklist100, gtItem)
        hits5.append(hr5)
        ndcgs5.append(ndcg5)
        hits10.append(hr10)
        ndcgs10.append(ndcg10)
        maps.append(ap)
        mrrs.append(mrr)
    final_hr5, final_ndcg5, final_hr10, final_ndcg10, final_map, final_mrr = np.array(
        hits5).mean(), np.array(ndcgs5).mean(), np.array(hits10).mean(), np.array(ndcgs10).mean(), np.array(
        maps).mean(), np.array(mrrs).mean()
    return (final_hr5, final_ndcg5, final_hr10, final_ndcg10, final_map, final_mrr, test_dict)


def init_test_dict():
    test_num_batch = 1289
    test_dict = {}
    for index in range(test_num_batch):
        map_course_score = {t: 0.0 for t in range(setting.TEST_BATCH_SIZE)}
        test_dict[index] = map_course_score

    return test_dict


def cal_mrr(model, sess, saver):
    valid_job, valid_course = generate_valid_data()
    eval_num_batch = int (len(valid_job) / setting.EVAL_BATCH_SIZE)
    saver.restore(sess, tf.train.get_checkpoint_state(
        os.path.dirname(setting.RANK_MODEL_CHECKPOINT_PATH + 'checkpoint')).model_checkpoint_path)
    print("calculate mrr, already load the bert_pointwise model...")
    mrrs = []
    for index in range(eval_num_batch):
        valid_batch_job, valid_batch_course, = generate_test_valid_batch_data(valid_job, valid_course, index)
        feed_dict = {model.job_input_id: valid_batch_job, model.course_input_id: valid_batch_course}
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

    # valid_job, valid_course = generate_valid_data()
    # eval_num_batch = len(valid_job)
    # print(eval_num_batch)
    # saver.restore(sess, tf.train.get_checkpoint_state(
    #     os.path.dirname(setting.RANK_MODEL_CHECKPOINT_PATH + 'checkpoint')).model_checkpoint_path)
    # print("calculate mrr, already load the ipo model...")

    valid_job, valid_course = generate_valid_data()
    valid_num_batch = int(len(valid_job) / setting.EVAL_BATCH_SIZE)
    print(valid_num_batch)
    saver.restore(sess, tf.train.get_checkpoint_state(
    os.path.dirname(setting.RANK_MODEL_CHECKPOINT_PATH + 'checkpoint')).model_checkpoint_path)


    job_course_dict = {}

    for index in range(valid_num_batch):
        valid_batch_job, valid_batch_course = generate_test_valid_batch_data(valid_job, valid_course, index)
        feed_dict = {model.job_input_id: valid_batch_job, model.course_input_id: valid_batch_course}
        per_batch_predict = sess.run([model.output], feed_dict)
        per_batch_predict = np.array(per_batch_predict).reshape(-1)
        temp = list(np.reshape(per_batch_predict, (-1)))
        map_course_score = {t: temp[t] for t in range(setting.TEST_BATCH_SIZE)}

        job_course_dict[index] = map_course_score

    return job_course_dict


def init_mrr_dict():
    valid_job, valid_course = generate_valid_data()
    eval_num_batch = len(valid_job)

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


