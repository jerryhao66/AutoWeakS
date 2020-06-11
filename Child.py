import tensorflow as tf
import numpy as np

from collections import Counter
import heapq

from Interaction_pointwise import ModelIPO
from Representation_pointwise import ModelRPO
from bert_association import run_classifier
from bert_association import evaluate as bert_evaluate
from GraphSAGE.graphsage import graphsage_evaluate
from GraphSAGE.graphsage import supervised_main_jeremyhao as ModelGraphSAGE
from bert_representation import brepr
import math

from GraphSAGE.graphsage.models import SampleAndAggregate, SAGEInfo, Node2VecModel
from GraphSAGE.graphsage.minibatch import EdgeMinibatchIterator
from GraphSAGE.graphsage.neigh_samplers import UniformNeighborSampler


class Child(object):
    def __init__(self):
        self.debug_arc = None
        self.num_unsuper_methods = 8
        self.num_super_methods = 5
        self.topk_adder = 20
        self.decision_sample = 1
        self.training_max_index = 566

        self.pseudo_training_file_rpo = './Representation_pointwise/standard_data/pairs/train_standard_positive_pairs_index.txt'
        self.pseudo_training_file_ipo = './Interaction_pointwise/standard_data/pairs/train_standard_positive_pairs_index.txt'
        self.pseudo_training_file_brpo = './bert_representation/standard_data/train_data/train_standard_positive_pairs_index.txt'
        self.pseudo_training_file_bipo = './bert_association/standard_data/train_data/train_standard_positive_pairs_index.txt'
        self.pseudo_training_file_graphsage = './GraphSAGE/graphsage/unsup-data/graphsage_meanpool_small_0.000010/standard_eval.txt'

        self.bm25_voting_dict = './majority_voting_dict/bm25_dict.txt'
        self.word2vec_voting_dict = './majority_voting_dict/metapath_dict.txt'
        self.bert_voting_dict = './majority_voting_dict/bertpre_dict.txt'
        self.line_voting_dict = './majority_voting_dict/line_dict.txt'
        self.pte_voting_dict = './majority_voting_dict/pte_dict.txt'
        self.deepwalk_voting_dict = './majority_voting_dict/deepwalk_dict.txt'
        self.node2vec_voting_dict = './majority_voting_dict/node2vec_dict.txt'
        self.graphsage_voting_dict = './majority_voting_dict/graphsage_dict.txt'

    def connect_controller(self, Controller, sample_arc_seqs):
        if self.debug_arc is None:
            self.sample_arcs = sample_arc_seqs
            # format of sample_arc_seqs
            # self.sample_arcs  = [[ np.array([1, 0, 0, 0, 0, 0, 0, 0]), np.array([0]), np.array([1, 1, 1, 1, 1, 1])]]
        else:
            self.sample_arcs = self.debug_arc

        self.sample_times = Controller.sample_times

        self._initialize_model()

        self._training_child_model_simple()

    def _initialize_model(self):

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        config = tf.ConfigProto(gpu_options=gpu_options)
        # config.gpu_options.allow_growth = True

        # Interaction pointwise model
        self.IPoint_graph = tf.Graph()
        self.IPoint_sess = tf.Session(graph=self.IPoint_graph, config=config)

        # Representation pointwise model
        self.RPoint_graph = tf.Graph()
        self.RPoint_sess = tf.Session(graph=self.RPoint_graph)

        # BERT representation
        self.BERTrepre_graph = tf.Graph()
        self.BERTrepre_sess = tf.Session(graph=self.BERTrepre_graph)

        # graphsage
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
        config.allow_soft_placement = True
        self.Graphsage_graph = tf.Graph()
        self.Graphsage_sess = tf.Session(graph=self.Graphsage_graph, config=config)

    def _training_child_model_simple(self):
        # print("Build training models combinations...")
        super_reward = []
        for index in range(self.sample_times):
            unsuper_choice, topk_choice, super_choice = self.sample_arcs[index][0], self.sample_arcs[index][1], \
                                                        self.sample_arcs[index][2]
            print(unsuper_choice, topk_choice, super_choice)

            string = ""
            for item in list(unsuper_choice):
                string += str(item)
            # print(string)

            # unsupervised methods
            topk_number = topk_choice[0] * 3 + self.topk_adder  # 3x + 20  [20, 23, 26, 29, 32]

            bm25_dict = self._read_final_dict_simple(self.bm25_voting_dict, topk_number)
            word2vec_dict = self._read_final_dict_simple(self.word2vec_voting_dict, topk_number)
            bertpre_dict = self._read_final_dict_simple(self.bert_voting_dict, topk_number)
            line_dict = self._read_final_dict_simple(self.line_voting_dict, topk_number)
            pte_dict = self._read_final_dict_simple(self.pte_voting_dict, topk_number)
            deepwalk_dict = self._read_final_dict_simple(self.deepwalk_voting_dict, topk_number)
            node2vec_dict = self._read_final_dict_simple(self.node2vec_voting_dict, topk_number)
            graphsage_dict = self._read_final_dict_simple(self.graphsage_voting_dict, topk_number)


            self._majority_voting_labels_plus_read_mrr_reward(bm25_dict, word2vec_dict, bertpre_dict, line_dict,
                                                              pte_dict,
                                                              deepwalk_dict, node2vec_dict, graphsage_dict, topk_number,
                                                              unsuper_choice, string)

            # supervised methods
            super_non_zero_index = [i for i in range(self.num_super_methods) if super_choice[i] == 1]
            reward, hr5, ndcg5, hr10, ndcg10, map, mrr = self._select_super_methods(super_non_zero_index)

            print('hr5 = %.4f, ndcg5 = %.4f, hr10 =%.4f, ndcg10= %.4f, map = %.4f' % (hr5, ndcg5, hr10, ndcg10, map))
            reward = self.unsuper_reward + reward
            super_reward.append(reward)

        self.rewards = np.array(super_reward)
        # print(self.rewards)

    def _select_super_methods(self, selected_list):
        if selected_list:
            for index in range(self.num_super_methods):
                if index in selected_list:
                    if index == 0:
                        # Representation pointwise
                        with self.RPoint_graph.as_default():
                            with self.RPoint_sess.as_default():
                                RPointmodel = ModelRPO.RPointModelNetwork()
                                self.RPoint_saver = tf.train.Saver()
                                self.RPoint_sess.run(tf.global_variables_initializer())
                                test_rpo_dict = ModelRPO.training(
                                    RPointmodel, self.RPoint_sess, self.RPoint_saver)
                                rpo_dict = ModelRPO.cal_mrr_dict(RPointmodel, self.RPoint_sess, self.RPoint_saver)

                    elif index == 1:
                        # Interaction pointwise
                        with self.IPoint_graph.as_default():
                            with self.IPoint_sess.as_default():
                                IPointmodel = ModelIPO.IPointModelNetwork()
                                self.IPoint_saver = tf.train.Saver()
                                self.IPoint_sess.run(tf.global_variables_initializer())
                                test_ipo_dict = ModelIPO.training(
                                    IPointmodel, self.IPoint_sess, self.IPoint_saver)

                                ipo_dict = ModelIPO.cal_mrr_dict(IPointmodel, self.IPoint_sess, self.IPoint_saver)
                    elif index == 2:
                        # Bert interaction model
                        run_classifier.main()
                        _, _, _, _, _, _, _, _, test_bert_dict = bert_evaluate.evaluate()
                        bert_dict = bert_evaluate.calculate_mrr()
                    elif index == 3:
                        # BERT representation model
                        with self.BERTrepre_graph.as_default():
                            with self.BERTrepre_sess.as_default():
                                BERTrepremodel = brepr.BERT_Representation()
                                self.BERTrepre_saver = tf.train.Saver()
                                self.BERTrepre_sess.run(tf.global_variables_initializer())
                                test_bert_dict1 = brepr.training(BERTrepremodel, self.BERTrepre_sess, self.BERTrepre_saver)
                                bert_dict1 = brepr.cal_mrr_dict(BERTrepremodel, self.BERTrepre_sess, self.BERTrepre_saver)

                    elif index == 4:
                        # GraphSAGE supervised
                        # tf.reset_default_graph()
                        with self.Graphsage_graph.as_default():
                            with self.Graphsage_sess.as_default():

                                # graphsagemodel = ModelGraphSAGE.generate_model()
                                # self.graphsage_saver = tf.train.Saver()
                                # ModelGraphSAGE.main_training(graphsagemodel, self.Graphsage_sess, self.graphsage_saver)
                                ModelGraphSAGE.main_training()
                                # print('cal test dict...')
                                test_graphsagesuper_dict = graphsage_evaluate.cal_test_dict()
                                hr5, ndcg5, hr20, ndcg20, hr10, ndcg10, map, mrr = graphsage_evaluate.evaluate()
                                graphsagesuper_dict = graphsage_evaluate.cal_mrr()

                else:
                    if index == 0:
                        # print('init representation pointwise performance and reward...')
                        test_rpo_dict = ModelRPO.init_test_dict()
                        rpo_dict = ModelRPO.init_mrr_dict()

                    elif index == 1:
                        # print('init interaction pointwise performance and reward...')
                        test_ipo_dict = ModelIPO.init_test_dict()
                        ipo_dict = ModelIPO.init_mrr_dict()

                    elif index == 2:
                        # print('init bert IR model and reward...')
                        test_bert_dict = bert_evaluate.init_test_dict()
                        bert_dict = bert_evaluate.init_mrr_dict()
                    elif index == 3:
                        # need to revise
                        # print('init bert IR model and reward...')
                        test_bert_dict1 = bert_evaluate.init_test_dict()
                        bert_dict1 = bert_evaluate.init_mrr_dict()

                    elif index == 4:
                        test_graphsagesuper_dict = graphsage_evaluate.init_test_dict()
                        graphsagesuper_dict = graphsage_evaluate.init_mrr_dict()

            assert test_ipo_dict.keys() == test_bert_dict.keys() == test_graphsagesuper_dict.keys() == test_bert_dict1.keys()
            f_hr5, f_ndcg5, _, _, f_hr10, f_ndcg10, f_map, f_mrr = self._cal_super_performance(test_ipo_dict,
                                                                                               test_rpo_dict,
                                                                                               test_bert_dict,
                                                                                               test_bert_dict1,
                                                                                               test_graphsagesuper_dict,
                                                                                               eval_num_batch=1289)

            _, _, _, _, _, _, _, super_reward = self._cal_super_performance(ipo_dict, rpo_dict,
                                                                            bert_dict, bert_dict1, graphsagesuper_dict,
                                                                            eval_num_batch=1400)

        else:
            # use Interaction pointwise as default
            with self.RPoint_graph.as_default():
                with self.RPoint_sess.as_default():
                    RPointmodel = ModelRPO.RPointModelNetwork()
                    self.RPoint_saver = tf.train.Saver()
                    self.RPoint_sess.run(tf.global_variables_initializer())

                    # training, restore the model to calculate reward
                    test_ipo_dict = ModelIPO.init_test_dict()
                    test_rpo_dict = ModelRPO.training(RPointmodel, self.RPoint_sess, self.RPoint_saver)
                    test_bert_dict = bert_evaluate.init_test_dict()
                    test_bert_dict1 = bert_evaluate.init_test_dict()
                    test_graphsagesuper_dict = graphsage_evaluate.init_test_dict()

                    f_hr5, f_ndcg5, _, _, f_hr10, f_ndcg10, f_map, f_mrr = self._cal_super_performance(test_ipo_dict,
                                                                                                       test_rpo_dict,
                                                                                                       test_bert_dict,
                                                                                                       test_bert_dict1,
                                                                                                       test_graphsagesuper_dict,
                                                                                                       eval_num_batch=1289)
                    # super_reward = ModelRPO.cal_reward(RPointmodel, self.RPoint_sess, self.RPoint_saver)
                    super_reward = ModelRPO.cal_mrr(RPointmodel, self.RPoint_sess, self.RPoint_saver)

        return super_reward, f_hr5, f_ndcg5, f_hr10, f_ndcg10, f_map, f_mrr

    def close_sess(self):
        self.IPoint_sess.close()
        self.RPoint_sess.close()
        self.BERTrepre_sess.close()

    def _cal_super_performance(self, ipo_dict, rpo_dict, bert_dict, bert_dict1, graphsage_dict, eval_num_batch):
        final_dict = {}
        for k in ipo_dict.keys():
            final_dict[k] = dict(
                Counter(ipo_dict[k]) + Counter(rpo_dict[k]) + Counter(
                    bert_dict[k]) + Counter(bert_dict1[k]) + Counter(graphsage_dict[k]))

        hits5, ndcgs5, hits10, ndcgs10, hits20, ndcgs20, maps, mrrs = [], [], [], [], [], [], [], []
        gtItem = 0
        for batch_index in range(eval_num_batch):
            map_course_score = final_dict[batch_index]
            ranklist5 = heapq.nlargest(5, map_course_score, key=map_course_score.get)
            ranklist10 = heapq.nlargest(10, map_course_score, key=map_course_score.get)
            ranklist20 = heapq.nlargest(20, map_course_score, key=map_course_score.get)
            ranklist100 = heapq.nlargest(100, map_course_score, key=map_course_score.get)
            hr5 = getHitRatio(ranklist5, gtItem)
            ndcg5 = getNDCG(ranklist5, gtItem)
            hr10 = getHitRatio(ranklist10, gtItem)
            ndcg10 = getNDCG(ranklist10, gtItem)
            hr20 = getHitRatio(ranklist20, gtItem)
            ndcg20 = getNDCG(ranklist20, gtItem)
            map = getNDCG(ranklist100, gtItem)
            mrr = getMRR(ranklist100, gtItem)
            hits5.append(hr5)
            ndcgs5.append(ndcg5)
            hits10.append(hr10)
            ndcgs10.append(ndcg10)
            hits20.append(hr20)
            ndcgs20.append(ndcg20)
            maps.append(map)

            mrrs.append(mrr)
        final_mrr = np.array(mrrs).mean()
        final_hr5, final_ndcg5, final_hr20, final_ndcg20, final_hr10, final_ndcg10, final_map, final_mrr = np.array(
            hits5).mean(), np.array(ndcgs5).mean(), np.array(hits20).mean(), np.array(
            ndcgs20).mean(), np.array(hits10).mean(), np.array(ndcgs10).mean(), np.array(maps).mean(), np.array(
            mrrs).mean()
        # return final_mrr
        return (final_hr5, final_ndcg5, final_hr20, final_ndcg20, final_hr10, final_ndcg10, final_map, final_mrr)

    def _read_original_dict_simple(self, original_dict, topk):
        final_dict = {}
        for key in original_dict.keys():
            for index in range(len(original_dict[key])):
                if index >= topk:
                    break
                else:
                    if key not in final_dict:
                        final_dict[key] = []
                        final_dict[key].append(original_dict[key][index])
                    else:
                        final_dict[key].append(original_dict[key][index])
        return final_dict

    def _read_final_dict_simple(self, file, topk):
        final_dict = {}
        with open(file, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(' ')
                job, course = int(arr[0]), arr[1:]
                for index in range(len(course)):
                    if index >= topk:
                        break
                    else:
                        if job not in final_dict:
                            final_dict[job] = []
                            final_dict[job].append(course[index])
                        else:
                            final_dict[job].append(course[index])

                line = f.readline()
        return final_dict

    def _majority_voting_labels_plus_read_mrr_reward(self, bm25_dict, word2vec_dict, bertpre_dict, line_dict, pte_dict,
                                                     deepwalk_dict, node2vec_dict, graphsage_dict, topk_number,
                                                     unsuper_choice, string):
        bm25_topk_dict = self._read_original_dict_simple(bm25_dict, topk_number)  # 0
        word2vec_topk_dict = self._read_original_dict_simple(word2vec_dict, topk_number)  # 1  
        bert_topk_dict = self._read_original_dict_simple(bertpre_dict, topk_number)  # 2
        line_topk_dict = self._read_original_dict_simple(line_dict, topk_number)  # 3
        pte_topk_dict = self._read_original_dict_simple(pte_dict, topk_number)  # 4
        deepwalk_topk_dict = self._read_original_dict_simple(deepwalk_dict, topk_number)  # 5  
        node2vec_topk_dict = self._read_original_dict_simple(node2vec_dict, topk_number)  # 6
        graphsage_topk_dict = self._read_original_dict_simple(graphsage_dict, topk_number)  # 7

        num_unsuper_method = len(unsuper_choice)
        select_unsuper_num = dict(Counter(unsuper_choice))[1]
        # print(dict(Counter(unsuper_choice)))

        majority_threshold_dict = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}

        voting_count_dict = {}
        for i in range(num_unsuper_method):
            if i == 0:
                if unsuper_choice[i] != 0:
                    for key in bm25_topk_dict.keys():
                        voting_count_dict[key] = {}
                        for item in bm25_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1
                else:
                    for key in bm25_topk_dict.keys():
                        voting_count_dict[key] = {}
                        for item in bm25_topk_dict[key]:
                            voting_count_dict[key][item] = 0

            elif i == 1:
                if unsuper_choice[i] != 0:
                    for key in word2vec_topk_dict.keys():
                        for item in word2vec_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1

            elif i == 2:
                if unsuper_choice[i] != 0:
                    for key in bert_topk_dict.keys():
                        for item in bert_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1
            elif i == 3:
                if unsuper_choice[i] != 0:
                    for key in line_topk_dict.keys():
                        for item in line_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1


            elif i == 4:
                if unsuper_choice[i] != 0:
                    for key in pte_topk_dict.keys():
                        for item in pte_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1

            elif i == 5:
                if unsuper_choice[i] != 0:
                    for key in deepwalk_topk_dict.keys():
                        for item in deepwalk_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1

            elif i == 6:
                if unsuper_choice[i] != 0:
                    for key in node2vec_topk_dict.keys():
                        for item in node2vec_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1


            elif i == 7:
                if unsuper_choice[i] != 0:
                    for key in graphsage_topk_dict.keys():
                        for item in graphsage_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1

        # write pseudo training files
        with open(self.pseudo_training_file_rpo, 'w') as writer1:
            with open(self.pseudo_training_file_ipo, 'w') as writer2:
                with open(self.pseudo_training_file_brpo, 'w') as writer3:
                    with open(self.pseudo_training_file_bipo, 'w') as writer4:
                        with open(self.pseudo_training_file_graphsage, 'w') as writer5:
                            for job_key in voting_count_dict.keys():
                                if int(job_key) >= self.training_max_index:
                                    continue
                                for item in voting_count_dict[job_key]:
                                    if voting_count_dict[job_key][item] >= majority_threshold_dict[select_unsuper_num]:
                                        writer1.write(str(job_key) + ' ' + str(item) + '\n')
                                        writer2.write(str(job_key) + ' ' + str(item) + '\n')
                                        writer3.write(str(job_key) + ' ' + str(item) + '\n')
                                        writer4.write(str(job_key) + ' ' + str(item) + '\n')
                                        writer5.write(str(job_key) + ' ' + str(item) + '\n')

        self.unsuper_reward = self._read_unsuper_eval_mrr_reward('./unsupervised_mrr/' + string)

    def _read_unsuper_eval_mrr_reward(self, file):
        with open(file, 'r') as f:
            line = f.readline()
            arr = line.strip().split(' ')
            reward = float(arr[0])
            return math.log(reward)


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