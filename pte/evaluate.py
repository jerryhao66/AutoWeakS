from ptealgo.setting import setting
import numpy as np
import math
import heapq


def cos_sim(c, j):
    '''
    calculate the cosine similarity between two vecotrs
    '''
    vector_a = np.mat(c)
    vector_b = np.mat(j)

    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


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
            for item in arr:
                per_instance_list.append(item)
            course_list.append(per_instance_list)
            line = f.readline()
    return course_list


def dot_product(c, j):  # course, job
    c = np.reshape(np.array(c), (-1))
    j = np.reshape(np.array(j), (-1))
    return np.dot(c, j)


def evaluate(vector, args):
    job_list = load_job_list(args.job_list)
    course_list = load_course_list(args.course_list)


    assert len(job_list) == len(course_list)

    hits5, ndcgs5, hits20, ndcgs20, hits10, ndcgs10, maps, mrrs = [], [], [], [], [], [], [], []
    for i in range(len(job_list)):
        predictions = []
        candidate_courses_list = course_list[i]
        job = job_list[i]
        job_ebd = vector[job]
        for item in candidate_courses_list:
            course_ebd = vector[item]
            predictions.append(cos_sim(course_ebd, job_ebd))
            # predictions.append(sigmoid(dot_product(course_ebd, job_ebd)))
        gtItem = candidate_courses_list[0]
        map_course_score = {candidate_courses_list[t]: predictions[t] for t in range(len(candidate_courses_list))}
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

    final_hr5, final_ndcg5, final_hr20, final_ndcg20, final_hr10, final_ndcg10, final_map, final_mrr = np.array(hits5).mean(), np.array(ndcgs5).mean(),np.array(hits20).mean(), np.array(
        ndcgs20).mean(), np.array(hits10).mean(), np.array(ndcgs10).mean(), np.array(maps).mean(), np.array(mrrs).mean()
    return (final_hr5, final_ndcg5, final_hr20, final_ndcg20, final_hr10, final_ndcg10, final_map, final_mrr)


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

    # if __name__ == '__main__':
    #     args = setting()
    #     job_list = load_job_list(args.job_list)
    #     course_list = load_course_list(args.course_list)