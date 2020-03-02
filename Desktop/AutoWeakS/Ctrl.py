__author__ = 'haobowen'

# version: one-epoch-update
import tensorflow as tf
__author__='haobowen'
import numpy as np
from Child import Child


def lstm(x, prev_c, prev_h, w):
    '''
    x: [1, self.lstm_size]
    prev_h: [1, self.lstm_size]
    w: [2*lstm_size, 4*lstm_size]
    i, f, o, g: [1, lstm_size]
    next_c next_h: [1, lstm_size]

    '''
    ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w)  # [1, 4lstm_size]
    i, f, o, g = tf.split(ifog, 4, axis=1)  # 4 * [1, lstm_size]
    i = tf.sigmoid(i)
    f = tf.sigmoid(f)
    o = tf.sigmoid(o)
    g = tf.tanh(g)
    next_c = i * g + f * prev_c
    next_h = o * tf.tanh(next_c)
    return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
    next_c, next_h = [], []
    # each lstm layer: vertical stack, so the function called stack lstm
    # number of lstm (stack) layers
    for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
        # stack lstm, the input of the second layer lstm is the h of the first layer of lstm
        inputs = x if layer_id == 0 else next_h[-1]
        curr_c, curr_h = lstm(inputs, _c, _h, _w)
        next_c.append(curr_c)
        next_h.append(curr_h)
    return next_c, next_h


class Controller(object):
    def __init__(self):
        self.name = "controller"

        self.lstm_num_layers = 2
        self.lstm_size = 32
        self.num_unsuper_methods = 8
        self.num_topk = 5
        self.num_super_methods = 5
        self.temperature = None
        self.tanh_constant = None
        self.sample_times = 1
        self.decision_sample = 1

        # previous reward
        self.prev_reward = -2.0

        # # compute gradient
        self.lr = 0.01

        self._create_params()
        self.tvars = tf.trainable_variables()
        # for item in self.tvars:
        #     print(item)
        self.train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="train_step")

        # gradient series
        self.tvars = tf.trainable_variables()

        # manual update parameters
        self.tvars_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.tvars_holders.append(placeholder)

        self.update_tvar_holder = []
        for idx, var in enumerate(self.tvars):
            update_tvar = tf.assign(var, self.tvars_holders[idx])
            self.update_tvar_holder.append(update_tvar)

        # update parameters using gradient
        self.gradient_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.update_batch = tf.train.AdagradOptimizer(self.lr).apply_gradients(zip(self.gradient_holders, self.tvars))

    def _create_params(self):
        initializer = tf.random_uniform_initializer(minval=-1.5, maxval=1.5)
        with tf.variable_scope(self.name, initializer=initializer):
            with tf.variable_scope("lstm"):
                self.w_lstm = []
                for layer_id in range(self.lstm_num_layers):
                    with tf.variable_scope("layer_{}".format(layer_id)):
                        w = tf.get_variable("w", [2 * self.lstm_size, 4 * self.lstm_size])
                        self.w_lstm.append(w)

            with tf.variable_scope("embedding"):
                self.s_ebd = tf.get_variable("start_ebd", shape=[1, self.lstm_size])
                # self.topk_ebd = tf.get_variable("topk_ebd", [self.num_topk, self.lstm_size])

            with tf.variable_scope("softmax"):
                self.w_soft = tf.get_variable("topk_softmax", [self.lstm_size, self.num_topk])

            with tf.variable_scope("attention"):
                # unsupervised methods
                self.w_l = tf.get_variable("line_weight", [self.lstm_size, self.lstm_size])
                self.w_p = tf.get_variable("pte_weight", [self.lstm_size, self.lstm_size])
                self.w_n = tf.get_variable("node2vec_weight", [self.lstm_size, self.lstm_size])
                self.w_m = tf.get_variable("metapath_weight", [self.lstm_size, self.lstm_size])
                self.w_d = tf.get_variable("deepwalk_weight", [self.lstm_size, self.lstm_size])
                self.w_b = tf.get_variable("bm25_weight", [self.lstm_size, self.lstm_size])
                self.w_graphsage = tf.get_variable("graphsage_weight", [self.lstm_size, self.lstm_size])
                self.w_bert_pre = tf.get_variable("bert_weight", [self.lstm_size, self.lstm_size])

                self.fc_unsuper = tf.get_variable("fully_connection_unsupervised", [self.lstm_size, 1])

                # supervised methods
                self.w_ipo = tf.get_variable("ipo_weight", [self.lstm_size, self.lstm_size])
                self.w_ipa = tf.get_variable("ipa_weight", [self.lstm_size, self.lstm_size])
                self.w_rpo = tf.get_variable("rpo_weight", [self.lstm_size, self.lstm_size])
                self.w_rpa = tf.get_variable("rpa_weight", [self.lstm_size, self.lstm_size])
                self.w_bert_fine = tf.get_variable("bert_weigth", [self.lstm_size, self.lstm_size])

                self.fc_super = tf.get_variable("fully_connection_supervised", [self.lstm_size, 1])

                # topk
                self.w_20 = tf.get_variable("topk20_weight", shape=[self.lstm_size, self.lstm_size])
                self.w_30 = tf.get_variable("topk30_weight", shape=[self.lstm_size, self.lstm_size])
                self.w_40 = tf.get_variable("topk40_weight", shape=[self.lstm_size, self.lstm_size])
                self.w_50 = tf.get_variable("topk50_weight", shape=[self.lstm_size, self.lstm_size])
                self.w_60 = tf.get_variable("topk60_weight", shape=[self.lstm_size, self.lstm_size])

    def build_sampler(self, mode):
        """Build the sampler ops and the log_prob ops."""
        if mode == 'train':
            sample_epoches = self.sample_times
        elif mode == 'eval':
            sample_epoches = self.decision_sample
        else:
            raise ValueError("Unknown value mode...")
        # print("start build controller sampler")
        sample_arc_seqs, sample_entropys, sample_log_probs = [], [], []

        for _ in range(sample_epoches):
            # supervised anchors, unsupervised anchors
            anchor = []

            arc_seq, entropys, log_probs = [], [], []

            prev_c = [tf.zeros([1, self.lstm_size], tf.float32) for _ in range(self.lstm_num_layers)]
            prev_h = [tf.zeros([1, self.lstm_size], tf.float32) for _ in range(self.lstm_num_layers)]

            inputs = self.s_ebd

            # unsupervised methods selection
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h

            # calculate next_h[-1] * all super to get attention weight, each get [1, lstm]
            anchor.append(tf.matmul(next_h[-1], self.w_l))
            anchor.append(tf.matmul(next_h[-1], self.w_p))
            anchor.append(tf.matmul(next_h[-1], self.w_n))
            anchor.append(tf.matmul(next_h[-1], self.w_m))
            anchor.append(tf.matmul(next_h[-1], self.w_d))
            anchor.append(tf.matmul(next_h[-1], self.w_b))
            anchor.append(tf.matmul(next_h[-1], self.w_graphsage))
            anchor.append(tf.matmul(next_h[-1], self.w_bert_pre))

            anchors = tf.identity(tf.concat(anchor, axis=0))  # [8, lstm]
            exp_anchors = tf.exp(anchors)
            sum_anchors = tf.reduce_sum(exp_anchors, 0, keep_dims=True)  # [8, lstm]
            w_atten = tf.div(exp_anchors, sum_anchors)  # [5, lstm]

            # calculate attention weight
            candidate_inputs = w_atten * anchors  # [8, lstm]

            # classification
            query = tf.matmul(candidate_inputs, self.fc_unsuper)
            logit = tf.concat([-query, query], axis=1)  # [8, 2]
            if self.temperature is not None:
                logit /= self.temperature
            if self.tanh_constant is not None:
                logit = self.tanh_constant * tf.tanh(logit)

            choice = tf.multinomial(logit, 1)  # [8, 1]
            choice = tf.to_int32(choice)  # [8, 1]
            choice = tf.reshape(choice, [self.num_unsuper_methods])  # [8,]

            arc_seq.append(choice)

            # gradient descent (unsupervised model)
            log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=choice)  # [batch_size, ]
            log_probs.append(tf.reduce_sum(log_prob, keep_dims=True))
            entropy = tf.stop_gradient(tf.reduce_sum(log_prob * tf.exp(-log_prob), keep_dims=True))
            entropys.append(entropy)

            # next input, use selected unsupervised embedding methods

            choice = tf.reshape(choice, [1, self.num_unsuper_methods])
            choice = tf.cast(choice, tf.float32)
            inputs = tf.matmul(choice, candidate_inputs)

            # topk
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h

            anchor = []
            anchor.append(tf.matmul(next_h[-1], self.w_20))
            anchor.append(tf.matmul(next_h[-1], self.w_30))
            anchor.append(tf.matmul(next_h[-1], self.w_40))
            anchor.append(tf.matmul(next_h[-1], self.w_50))
            anchor.append(tf.matmul(next_h[-1], self.w_60))

            anchors = tf.identity(tf.concat(anchor, axis=0))
            exp_anchors = tf.exp(anchors)
            sum_anchors = tf.reduce_sum(exp_anchors, 0, keep_dims=True)  # [1, lstm]
            w_atten = tf.div(exp_anchors, sum_anchors)
            candidate_inputs = w_atten * anchors

            for i in range(1):
                input_list = []
                logit = tf.matmul(next_h[-1], self.w_soft)
                if self.temperature is not None:
                    logit /= self.temperature
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * tf.tanh(logit)
                branch_id = tf.multinomial(logit, 1)
                branch_id = tf.to_int32(branch_id)
                branch_id = tf.reshape(branch_id, [1])
                self.branch_id = branch_id

                arc_seq.append(branch_id)
                classes = self.num_topk
                self.mask_input = tf.one_hot(branch_id, classes)

                # next input
                inputs = tf.matmul(self.mask_input, candidate_inputs)
                input_list.append(inputs)

                # gradient desecent (topk)
                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=branch_id)
                log_probs.append(log_prob)
                entropy = tf.stop_gradient(tf.reduce_sum(log_prob * tf.exp(-log_prob), keep_dims=True))
                entropys.append(entropy)

            inputs = tf.concat(input_list, 0)
            # supervised methods selection
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)

            anchor = []
            anchor.append(tf.matmul(next_h[-1], self.w_ipo))
            anchor.append(tf.matmul(next_h[-1], self.w_ipa))
            anchor.append(tf.matmul(next_h[-1], self.w_rpo))
            anchor.append(tf.matmul(next_h[-1], self.w_rpa))
            anchor.append(tf.matmul(next_h[-1], self.w_bert_fine))

            anchors = tf.identity(tf.concat(anchor, axis=0))
            exp_anchors = tf.exp(anchors)
            sum_anchors = tf.reduce_sum(exp_anchors, 0, keep_dims=True)
            w_atten = tf.div(exp_anchors, sum_anchors)  # [5, lstm]

            # calculate attention weight
            candidate_inputs = w_atten * anchors  # [5, lstm]

            # classification
            query = tf.matmul(candidate_inputs, self.fc_super)
            logit = tf.concat([-query, query], axis=1)

            if self.temperature is not None:
                logit /= self.temperature
            if self.tanh_constant is not None:
                logit = self.tanh_constant * tf.tanh(logit)

            choice = tf.multinomial(logit, 1)  # [5, 1]
            choice = tf.to_int32(choice)  # [5, 1]
            choice = tf.reshape(choice, [self.num_super_methods])  # [5,]

            arc_seq.append(choice)

            # gradient descent (unsupervised model)
            log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=choice)  # [batch_size, ]
            log_probs.append(tf.reduce_sum(log_prob, keep_dims=True))
            entropy = tf.stop_gradient(tf.reduce_sum(log_prob * tf.exp(-log_prob), keep_dims=True))
            entropys.append(entropy)

            sample_arc_seqs.append(arc_seq)
            sample_entropys.append(entropys)
            sample_log_probs.append(log_probs)

        self.sample_arc_seqs, self.sample_entropys, self.sample_log_probs = sample_arc_seqs, sample_entropys, sample_log_probs

    def upgrade_network(self, sess, child_model):
        updaterate = 0.8
        tvars_best = sess.run(self.tvars)
        for index, var in enumerate(tvars_best):
            tvars_best[index] = var * 0
        tvars_old = sess.run(self.tvars)

        gradBuffer = sess.run(self.tvars)
        for index, grad in enumerate(gradBuffer):
            gradBuffer[index] = grad * 0

        rewards = child_model.rewards

        baseline = self.prev_reward

        for index in range(self.sample_times):
            reward = rewards[index]

            sample_log_prob = self.sample_log_probs[index]
            sample_log_prob = tf.reduce_sum(sample_log_prob)
            self.loss = sample_log_prob * (reward - baseline)
            # compute gradient
            grads = sess.run(tf.gradients(self.loss, self.tvars))

            for index, grad in enumerate(grads):

                gradBuffer[index] += grad

        # apply gradient
        feed_dict = dictionary = dict(zip(self.gradient_holders, gradBuffer))
        sess.run(self.update_batch, feed_dict=feed_dict)
        for index, grad in enumerate(gradBuffer):
            gradBuffer[index] = grad * 0

        # get tvars_new
        tvars_new = sess.run(self.tvars)

        # update old variables of the target network
        tvars_update = sess.run(self.tvars)
        for index, var in enumerate(tvars_update):
            tvars_update[index] = updaterate * tvars_new[index] + (1 - updaterate) * tvars_old[index]

        feed_dict = dictionary = dict(zip(self.tvars_holders, tvars_update))
        sess.run(self.update_tvar_holder, feed_dict)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    Ctrl_graph = tf.Graph()
    Ctrl_sess = tf.Session(graph=Ctrl_graph, config=config)

    with Ctrl_graph.as_default():
        with Ctrl_sess.as_default():
            Controller = Controller()
            Ctrl_saver = tf.train.Saver()
            Ctrl_sess.run(tf.global_variables_initializer())
            Controller.build_sampler(mode='train')
            sample_arc_seqs = Ctrl_sess.run(Controller.sample_arc_seqs)

    child = Child()
    child.connect_controller(Controller, sample_arc_seqs)

    with Ctrl_graph.as_default():
        with Ctrl_sess.as_default():
            Controller.upgrade_network(Ctrl_sess, child)  
            Controller.build_sampler(mode='train')
            sample_arc_seqs = Ctrl_sess.run(Controller.sample_arc_seqs)

    Controller.prev_reward = np.mean(child.rewards)
    child.connect_controller(Controller, sample_arc_seqs)
    child.close_sess()

    with Ctrl_graph.as_default():
        with Ctrl_sess.as_default():
            Controller.upgrade_network(Ctrl_sess, child) 
             
    # loop
    for epoch in range(1):
        with Ctrl_graph.as_default():
            with Ctrl_sess.as_default():
                Controller.build_sampler(mode='train')
                sample_arc_seqs = Ctrl_sess.run(Controller.sample_arc_seqs)


        Controller.prev_reward = np.mean(child.rewards)
        print('epoch %d, reward is %.6f' % (epoch, Controller.prev_reward))
        child.connect_controller(Controller, sample_arc_seqs)
        child.close_sess()

        with Ctrl_graph.as_default():
            with Ctrl_sess.as_default():
                Controller.upgrade_network(Ctrl_sess, child)  

    Ctrl_sess.close()

