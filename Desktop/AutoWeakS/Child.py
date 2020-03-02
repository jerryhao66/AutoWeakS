__author__ = 'haobowen'

import tensorflow as tf
import numpy as np

from collections import Counter
import heapq

from Representation_pairwise import ModelRPA
from Interaction_pairwise import ModelIPA
from Interaction_pointwise import ModelIPO
from Representation_pointwise import ModelRPO
from bert_association import run_classifier
from bert_association import evaluate as bert_evaluate
import math


class Child(object):
    def __init__(self):
        self.debug_arc = None
        self.num_unsuper_methods = 5
        self.num_super_methods = 5
        self.topk_adder = 20
        self.decision_sample = 1
        self.training_max_index = 566

    def connect_controller(self, Controller, sample_arc_seqs):
        if self.debug_arc is None:
            self.sample_arcs = sample_arc_seqs
            # format of sample_arc_seqs
            # self.sample_arcs  = [[ np.array([1, 0, 0, 0, 0, 0, 0, 0]), np.array([0]), np.array([1, 1, 1, 1, 1])]]
        else:
            self.sample_arcs = self.debug_arc

        self.sample_times = Controller.sample_times

        self._initialize_model()

        self._training_child_model_simple()


    def _initialize_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Interaction pointwise model
        self.IPoint_graph = tf.Graph()
        self.IPoint_sess = tf.Session(graph=self.IPoint_graph, config=config)

        # Interaction pairwise model
        self.IPair_graph = tf.Graph()
        self.IPair_sess = tf.Session(graph=self.IPair_graph, config=config)

        # Representation pairwise model
        self.RPair_graph = tf.Graph()
        self.RPair_sess = tf.Session(graph=self.RPair_graph, config=config)

        # Representation pointwise model
        self.RPoint_graph = tf.Graph()
        self.RPoint_sess = tf.Session(graph=self.RPoint_graph)


    def _training_child_model_simple(self):
        # print("Build training models combinations...")
        super_reward = []
        for index in range(self.sample_times):
            unsuper_choice, topk_choice, super_choice = self.sample_arcs[index][0], self.sample_arcs[index][1], self.sample_arcs[index][2]
            print(unsuper_choice, topk_choice, super_choice)

            string = ""
            for item in list(unsuper_choice):
                string += str(item)


            # unsupervised methods
            topk_number = topk_choice[0] * 3 + self.topk_adder  # 3x + 20  [20, 23, 26, 29, 32]
          

            line_dict = self._read_final_dict_simple('./majority_voting_dict/line_dict.txt', topk_number)
            pte_dict = self._read_final_dict_simple('./majority_voting_dict/pte_dict.txt', topk_number)
            node2vec_dict = self._read_final_dict_simple('./majority_voting_dict/node2vec_dict.txt', topk_number)
            metapath_dict = self._read_final_dict_simple('./majority_voting_dict/metapath_dict.txt', topk_number)
            deepwalk_dict = self._read_final_dict_simple('./majority_voting_dict/deepwalk_dict.txt', topk_number)
            bm25_dict = self._read_final_dict_simple('./majority_voting_dict/bm25_dict.txt', topk_number)
            graphsage_dict = self._read_final_dict_simple('./majority_voting_dict/graphsage_dict.txt', topk_number)
            bertpre_dict = self._read_final_dict_simple('./majority_voting_dict/bertpre_dict.txt', topk_number)

            self._majority_voting_labels_plus_read_mrr_reward(line_dict, pte_dict, node2vec_dict, metapath_dict,
                                                              deepwalk_dict, bm25_dict, graphsage_dict, bertpre_dict,
                                                              topk_number, unsuper_choice, string)

            # supervised methods
            super_non_zero_index = [i for i in range(self.num_super_methods) if super_choice[i] == 1]
            reward, hr5, ndcg5, hr10, ndcg10, map, mrr = self._select_super_methods(super_non_zero_index)
    
            print('hr5 = %.4f, ndcg5 = %.4f, hr10 =%.4f, ndcg10= %.4f, map = %.4f' % (hr5, ndcg5, hr10, ndcg10, map))
            reward = self.unsuper_reward + reward
            super_reward.append(reward)
    
    
        self.rewards = np.array(super_reward)


    def _select_super_methods(self, selected_list):
        if selected_list:
            for index in range(self.num_super_methods):
                if index in selected_list:
                    if index == 0:
                        # Interaction pointwise
                        with self.IPoint_graph.as_default():
                            with self.IPoint_sess.as_default():
                                IPointmodel = ModelIPO.IPointModelNetwork()
                                self.IPoint_saver = tf.train.Saver()
                                self.IPoint_sess.run(tf.global_variables_initializer())
                                test_ipo_dict = ModelIPO.training(
                                    IPointmodel, self.IPoint_sess, self.IPoint_saver)
                               
                                ipo_dict = ModelIPO.cal_mrr_dict(IPointmodel, self.IPoint_sess, self.IPoint_saver)
                    elif index == 1:
                        # Interaction pairwise
                        with self.IPair_graph.as_default():
                            with self.IPair_sess.as_default():
                                IPairmodel = ModelIPA.IPairModelNetwork()
                                self.IPair_saver = tf.train.Saver()
                                self.IPair_sess.run(tf.global_variables_initializer())
                                test_ipa_dict = ModelIPA.training(
                                    IPairmodel, self.IPair_sess, self.IPair_saver)
                             
                                ipa_dict = ModelIPA.cal_mrr_dict(IPairmodel, self.IPair_sess, self.IPair_saver)
                    elif index == 2:
                        # Representation pointwise
                        with self.RPoint_graph.as_default():
                            with self.RPoint_sess.as_default():
                                RPointmodel = ModelRPO.RPointModelNetwork()
                                self.RPoint_saver = tf.train.Saver()
                                self.RPoint_sess.run(tf.global_variables_initializer())
                                test_rpo_dict = ModelRPO.training(
                                    RPointmodel, self.RPoint_sess, self.RPoint_saver)
                                rpo_dict = ModelRPO.cal_mrr_dict(RPointmodel, self.RPoint_sess, self.RPoint_saver)
    
                    elif index == 3:
                        # Representation pairwise
                        with self.RPair_graph.as_default():
                            with self.RPair_sess.as_default():
                                RPairmodel = ModelRPA.RPairModelNetwork()
                                self.RPair_saver = tf.train.Saver()
                                self.RPair_sess.run(tf.global_variables_initializer())
                                test_rpa_dict = ModelRPA.training(
                                    RPairmodel, self.RPair_sess, self.RPair_saver)
                                rpa_dict = ModelRPA.cal_mrr_dict(RPairmodel, self.RPair_sess, self.RPair_saver)
                    elif index == 4:
                        # Bert IR model
                        run_classifier.main()
                        _, _, _, _, _, _, _, _, test_bert_dict = bert_evaluate.evaluate()
                        bert_dict = bert_evaluate.calculate_mrr()
    
                else:
                    if index == 0:
                        # print('init interaction pointwise performance and reward...')
                        test_ipo_dict = ModelIPO.init_test_dict()
    
                        ipo_dict = ModelIPO.init_mrr_dict()
    
                    elif index == 1:
                        # print('init interaction pairwise performance and reward...')
                        test_ipa_dict = ModelIPA.init_test_dict()
    
                        ipa_dict = ModelIPA.init_mrr_dict()
    
                    elif index == 2:
                        # print('init representation pointwise performance and reward...')
                        test_rpo_dict = ModelRPO.init_test_dict()
    
                        rpo_dict = ModelRPO.init_mrr_dict()
    
                    elif index == 3:
                        # print('init representation pairtwise performance and reward...')
                        test_rpa_dict = ModelRPA.init_test_dict()
    
                        rpa_dict = ModelRPA.init_mrr_dict()
                    elif index == 4:
                        # print('init bert IR model and reward...')
                        test_bert_dict = bert_evaluate.init_test_dict()
                        bert_dict = bert_evaluate.init_mrr_dict()
    
            f_hr5, f_ndcg5, _, _, f_hr10, f_ndcg10, f_map, f_mrr = self._cal_super_performance(test_ipo_dict, test_ipa_dict,
                                                                                               test_rpo_dict, test_rpa_dict,
                                                                                               test_bert_dict,
                                                                                               eval_num_batch=1289)
    
            _, _, _, _, _, _, _, super_reward = self._cal_super_performance(ipo_dict, ipa_dict, rpo_dict, rpa_dict,
                                                                            bert_dict, eval_num_batch=1400)

        else:
            # use Interaction pointwise as default
            with self.RPoint_graph.as_default():
                with self.RPoint_sess.as_default():
                    RPointmodel = ModelRPO.RPointModelNetwork()
                    self.RPoint_saver = tf.train.Saver()
                    self.RPoint_sess.run(tf.global_variables_initializer())
    
                    # training, restore the model to calculate reward
                    test_ipo_dict = ModelIPO.init_test_dict()
                    test_ipa_dict = ModelIPA.init_test_dict()
                    test_rpo_dict = ModelRPO.training(RPointmodel, self.RPoint_sess, self.RPoint_saver)
                    test_rpa_dict = ModelRPA.init_test_dict()
                    test_bert_dict = bert_evaluate.init_test_dict()
    
                    f_hr5, f_ndcg5, _, _, f_hr10, f_ndcg10, f_map, f_mrr = self._cal_super_performance(test_ipo_dict,
                                                                                                       test_ipa_dict,
                                                                                                       test_rpo_dict,
                                                                                                       test_rpa_dict,
                                                                                                       test_bert_dict,
                                                                                                       eval_num_batch=1289)
                    # super_reward = ModelRPO.cal_reward(RPointmodel, self.RPoint_sess, self.RPoint_saver)
                    super_reward = ModelRPO.cal_mrr(RPointmodel, self.RPoint_sess, self.RPoint_saver)
    
        return super_reward, f_hr5, f_ndcg5, f_hr10, f_ndcg10, f_map, f_mrr

    def close_sess(self):
        self.IPoint_sess.close()
        self.IPair_sess.close()
        self.RPair_sess.close()
        self.RPoint_sess.close()

    def _cal_super_performance(self, ipo_dict, ipa_dict, rpo_dict, rpa_dict, bert_dict, eval_num_batch):
        final_dict = {}
        for k in ipo_dict.keys():
            final_dict[k] = dict(
                Counter(ipo_dict[k]) + Counter(ipa_dict[k]) + Counter(rpo_dict[k]) + Counter(rpa_dict[k]) + Counter(
                    bert_dict[k]))
    
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

    def _majority_voting_labels_plus_read_mrr_reward(self, line_dict, pte_dict, node2vec_dict, metapath_dict,
                                                     deepwalk_dict, bm25_dict, graphsage_dict, bertpre_dict, topk,
                                                    unsuper_choice, string):
        line_topk_dict = self._read_original_dict_simple(line_dict, topk)  # 0
        pte_topk_dict = self._read_original_dict_simple(pte_dict, topk)  # 1
        node2vec_topk_dict = self._read_original_dict_simple(node2vec_dict, topk)  # 2
        metapath_topk_dict = self._read_original_dict_simple(metapath_dict, topk)  # 3
        deepwalk_topk_dict = self._read_original_dict_simple(deepwalk_dict, topk)  # 4
        bm25_topk_dict = self._read_original_dict_simple(bm25_dict, topk)  # 5
        graphsage_topk_dict = self._read_original_dict_simple(graphsage_dict, topk)  # 6
        bert_topk_dict = self._read_original_dict_simple(bertpre_dict, topk)  # 7

        num_unsuper_method = len(unsuper_choice)
        select_unsuper_num = dict(Counter(unsuper_choice))[1]
        # print(dict(Counter(unsuper_choice)))

        majority_threshold_dict = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}

        voting_count_dict = {}
        for i in range(num_unsuper_method):
            if i == 0:
                if unsuper_choice[i] != 0:
                    for key in line_topk_dict.keys():
                        voting_count_dict[key] = {}
                        for item in line_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1
                else:
                    for key in line_topk_dict.keys():
                        voting_count_dict[key] = {}
                        for item in line_topk_dict[key]:
                            voting_count_dict[key][item] = 0
            elif i == 1:
                if unsuper_choice[i] != 0:
                    for key in pte_topk_dict.keys():
                        for item in pte_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1

            elif i == 2:
                if unsuper_choice[i] != 0:
                    for key in node2vec_topk_dict.keys():
                        for item in node2vec_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1
            elif i == 3:
                if unsuper_choice[i] != 0:
                    for key in metapath_topk_dict.keys():
                        for item in metapath_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1
            elif i == 4:
                if unsuper_choice[i] != 0:
                    for key in deepwalk_topk_dict.keys():
                        for item in deepwalk_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1
            elif i == 5:
                if unsuper_choice[i] != 0:
                    for key in bm25_topk_dict.keys():
                        for item in bm25_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1
            elif i == 6:
                if unsuper_choice[i] != 0:
                    for key in graphsage_topk_dict.keys():
                        for item in graphsage_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1
            elif i == 7:
                if unsuper_choice[i] != 0:
                    for key in bert_topk_dict.keys():
                        for item in bert_topk_dict[key]:
                            if item not in voting_count_dict[key]:
                                voting_count_dict[key][item] = 1
                            else:
                                voting_count_dict[key][item] += 1

        # write pseudo training files

        with open('./Interaction_pairwise/standard_data/pairs/train_standard_positive_pairs_index.txt',
                  'w') as writer1:
            with open('./Interaction_pointwise/standard_data/pairs/train_standard_positive_pairs_index.txt',
                      'w') as writer2:
                with open(
                        './Representation_pairwise/standard_data/pairs/train_standard_positive_pairs_index.txt',
                        'w') as writer3:
                    with open(
                            './Representation_pointwise/standard_data/pairs/train_standard_positive_pairs_index.txt',
                            'w') as writer4:
                        with open(
                                './bert_association/standard_data/train_data/train_standard_positive_pairs_index.txt',
                                'w') as writer5:
                            for job_key in voting_count_dict.keys():
                                if int(job_key) >= self.training_max_index:
                                    continue
                                for item in voting_count_dict[job_key]:
                                    if voting_count_dict[job_key][item] >= majority_threshold_dict[
                                        select_unsuper_num]:
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