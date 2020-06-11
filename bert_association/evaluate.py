import heapq
import math
import numpy as np
from bert_association import setting
import os

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

def evaluate():

    all_prediction = []
    with open(os.path.join(setting.output_dir, "test_results.tsv"), 'r') as f:
    # with open('./output/test_results.tsv', 'r') as f:
        line = f.readline()
        while line!="" and line!=None:
            arr = line.strip().split('\t')
            zero_class_score, one_class_score = float(arr[0]), float(arr[1])
            all_prediction.append(one_class_score)
            line = f.readline()
    hits1, hits3, hits5, ndcgs1, ndcgs3, ndcgs5, hits20, ndcgs20, hits10, ndcgs10, maps, mrrs = [], [], [], [], [], [], [], [], [], [], [], []
    test_dict = {}
    num_batch = int( len(all_prediction) / 100)

    for index in range(num_batch):
        predictions = all_prediction[index * 100: index * 100 + 100]

        gtItem = 0
        map_course_score = {t: predictions[t] for t in range(100)}
        test_dict[index] = map_course_score
        # ranklist1 = heapq.nlargest(1, map_course_score, key=map_course_score.get)
        # ranklist3 = heapq.nlargest(3, map_course_score, key=map_course_score.get)
        ranklist5 = heapq.nlargest(5, map_course_score, key=map_course_score.get)
        ranklist20 = heapq.nlargest(20, map_course_score, key=map_course_score.get)
        ranklist10 = heapq.nlargest(10, map_course_score, key=map_course_score.get)
        ranklist100 = heapq.nlargest(100, map_course_score, key=map_course_score.get)
        # hr1 = getHitRatio(ranklist1, gtItem)
        # hr3 = getHitRatio(ranklist3, gtItem)
        hr5 = getHitRatio(ranklist5, gtItem)
        # ndcg1 = getNDCG(ranklist1, gtItem)
        # ndcg3 = getNDCG(ranklist3, gtItem)
        ndcg5 = getNDCG(ranklist5, gtItem)
        hr20 = getHitRatio(ranklist20, gtItem)
        ndcg20 = getNDCG(ranklist20, gtItem)
        hr10 = getHitRatio(ranklist10, gtItem)
        ndcg10 = getNDCG(ranklist10, gtItem)
        ap = getAP(ranklist100, gtItem)
        mrr = getMRR(ranklist100, gtItem)
        # hits1.append(hr1)
        # hits3.append(hr3)
        hits5.append(hr5)
        # ndcgs1.append(ndcg1)
        # ndcgs3.append(ndcg3)
        ndcgs5.append(ndcg5)
        hits20.append(hr20)
        ndcgs20.append(ndcg20)
        hits10.append(hr10)
        ndcgs10.append(ndcg10)
        maps.append(ap)
        mrrs.append(mrr)

    final_hr5, final_hr10, final_hr20, final_ndcg5, final_ndcg10, final_ndcg20, final_map, final_mrr = \
        np.array(hits5).mean(), np.array(hits10).mean(), np.array(
        hits20).mean(), np.array(ndcgs5).mean(), np.array(
        ndcgs10).mean(), np.array(ndcgs20).mean(), np.array(maps).mean(), np.array(mrrs).mean()
    return (final_hr5, final_hr10, final_hr20, final_ndcg5, final_ndcg10,
            final_ndcg20, final_map, final_mrr, test_dict)


    # final_hr1, final_hr3, final_hr5, final_hr10, final_hr20, final_ndcg1, final_ndcg3, final_ndcg5, final_ndcg10, final_ndcg20, final_map, final_mrr = \
    #     np.array(
    #     hits1).mean(), np.array(hits3).mean(), np.array(hits5).mean(), np.array(hits10).mean(), np.array(
    #     hits20).mean(), np.array(ndcgs1).mean(), np.array(ndcgs3).mean(), np.array(ndcgs5).mean(), np.array(
    #     ndcgs10).mean(), np.array(ndcgs20).mean(), np.array(maps).mean(), np.array(mrrs).mean()
    # return (final_hr1, final_hr3, final_hr5, final_hr10, final_hr20, final_ndcg1, final_ndcg3, final_ndcg5, final_ndcg10,
    #         final_ndcg20, final_map, final_mrr)

def calculate_mrr():
    all_prediction = []
    with open(os.path.join(setting.output_dir, "eval_results.tsv"), 'r') as f:
    # with open('./output/eval_results.csv', 'r') as f:
        line = f.readline()
        while line!="" and line!=None:
            arr = line.strip().split('\t')
            zero_class_score, one_class_score = float(arr[0]), float(arr[1])
            all_prediction.append(one_class_score)
            line = f.readline()

    EVAL_BATCH_SIZE = 100
    eval_num_batch = int(len(all_prediction) / EVAL_BATCH_SIZE)

    job_course_dict = {}

    for index in range(eval_num_batch):
        per_batch_predict = all_prediction[index * EVAL_BATCH_SIZE: index * EVAL_BATCH_SIZE+EVAL_BATCH_SIZE]
        map_course_score = {t: per_batch_predict[t] for t in range(EVAL_BATCH_SIZE)}
        job_course_dict[index] = map_course_score

    return job_course_dict

def init_mrr_dict():
    all_prediction = []
    with open(os.path.join(setting.output_dir, "eval_results.tsv"), 'r') as f:
    # with open('./output/eval_results.csv', 'r') as f:
        line = f.readline()
        while line!="" and line!=None:
            arr = line.strip().split('\t')
            zero_class_score, one_class_score = float(arr[0]), float(arr[1])
            all_prediction.append(one_class_score)
            line = f.readline()
    EVAL_BATCH_SIZE = 100
    eval_num_batch = int(len(all_prediction) / EVAL_BATCH_SIZE)

    job_course_dict = {}

    for index in range(eval_num_batch):
        map_course_score = {t: 0.0 for t in range(EVAL_BATCH_SIZE)}

        job_course_dict[index] = map_course_score

    return job_course_dict

def init_test_dict():
    TEST_BATCH_SIZE = 100
    test_num_batch = 1289
    test_dict = {}
    for index in range(test_num_batch):
        map_course_score = {t: 0.0 for t in range(TEST_BATCH_SIZE)}
        test_dict[index] = map_course_score

    return test_dict

if __name__ == '__main__':

    hr1, hr3, hr5, hr10, hr20, ndcg1, ndcg3, ndcg5, ndcg10, ndcg20, map, mrr = evaluate()
    print(
        'bert finetuning score is  hr1 = %.4f, hr3 = %.4f, hr5 = %.4f, hr10 = %.4f, hr20 = %.4f, ndcg1 = %.4f, ndcg3 = %.4f, ndcg5 = %.4f, ndcg10 = %.4f, ndcg20 = %.4f, map = %.4f, mrr = %.4f ' %
        (hr1, hr3, hr5, hr10, hr20, ndcg1, ndcg3, ndcg5, ndcg10, ndcg20, map, mrr))

