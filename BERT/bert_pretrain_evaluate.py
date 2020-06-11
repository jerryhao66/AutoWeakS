import numpy as np
import heapq
import math

global TEST_FILE
global COURSE_EBD
global JOB_EBD
global BATCH_SIZE

TEST_FILE = './test_pointwise.txt'
COURSE_EBD = np.load('./bert_pretrain_course.npy')
JOB_EBD = np.load('./bert_pretrain_job.npy')
BATCH_SIZE = 100


def load_job_course(file):
    job_list, course_list = [], []
    with open(file, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            job, course = int(arr[0]), int(arr[1])
            job_list.append(job)
            course_list.append(course)
            line = f.readline()
    return job_list, course_list


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


def evaluate():
    all_job, all_course = load_job_course(TEST_FILE)
    assert len(all_job) == len(all_course)

    hits1, hits3, hits5, ndcgs1, ndcgs3, ndcgs5, hits20, ndcgs20, hits10, ndcgs10, maps, mrrs = [], [], [], [], [], [], [], [], [], [], [], []

    num_batch = int(len(all_job) / BATCH_SIZE)

    for i in range(num_batch):
        job_list = all_job[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE]
        course_list = all_course[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE]

        predictions = []

        job_ebd = np.array(JOB_EBD[job_list[0]]).astype(np.float32)
        for item in course_list:
            course_ebd = np.array(COURSE_EBD[item]).astype(np.float32)
            predictions.append(cos_sim(course_ebd, job_ebd))

        # predictions.append(sigmoid(dot_product(course_ebd, job_ebd)))
        gtItem = course_list[0]
        map_course_score = {course_list[t]: predictions[t] for t in range(len(course_list))}
        ranklist1 = heapq.nlargest(1, map_course_score, key=map_course_score.get)
        ranklist3 = heapq.nlargest(3, map_course_score, key=map_course_score.get)
        ranklist5 = heapq.nlargest(5, map_course_score, key=map_course_score.get)
        ranklist20 = heapq.nlargest(20, map_course_score, key=map_course_score.get)
        ranklist10 = heapq.nlargest(10, map_course_score, key=map_course_score.get)
        ranklist100 = heapq.nlargest(100, map_course_score, key=map_course_score.get)
        hr1 = getHitRatio(ranklist1, gtItem)
        hr3 = getHitRatio(ranklist3, gtItem)
        hr5 = getHitRatio(ranklist5, gtItem)
        ndcg1 = getNDCG(ranklist1, gtItem)
        ndcg3 = getNDCG(ranklist3, gtItem)
        ndcg5 = getNDCG(ranklist5, gtItem)
        hr20 = getHitRatio(ranklist20, gtItem)
        ndcg20 = getNDCG(ranklist20, gtItem)
        hr10 = getHitRatio(ranklist10, gtItem)
        ndcg10 = getNDCG(ranklist10, gtItem)
        ap = getAP(ranklist100, gtItem)
        mrr = getMRR(ranklist100, gtItem)
        hits1.append(hr1)
        hits3.append(hr3)
        hits5.append(hr5)
        ndcgs1.append(ndcg1)
        ndcgs3.append(ndcg3)
        ndcgs5.append(ndcg5)
        hits20.append(hr20)
        ndcgs20.append(ndcg20)
        hits10.append(hr10)
        ndcgs10.append(ndcg10)
        maps.append(ap)
        mrrs.append(mrr)


    final_hr1, final_hr3, final_hr5, final_hr10, final_hr20, final_ndcg1, final_ndcg3, final_ndcg5, final_ndcg10, final_ndcg20, final_map, final_mrr = \
        np.array(
        hits1).mean(), np.array(hits3).mean(), np.array(hits5).mean(), np.array(hits10).mean(), np.array(
        hits20).mean(), np.array(ndcgs1).mean(), np.array(ndcgs3).mean(), np.array(ndcgs5).mean(), np.array(
        ndcgs10).mean(), np.array(ndcgs20).mean(), np.array(maps).mean(), np.array(mrrs).mean()
    return (final_hr1, final_hr3, final_hr5, final_hr10, final_hr20, final_ndcg1, final_ndcg3, final_ndcg5, final_ndcg10,
            final_ndcg20, final_map, final_mrr)


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

if __name__ == '__main__':
    hr1, hr3, hr5, hr10, hr20, ndcg1, ndcg3, ndcg5, ndcg10, ndcg20, map, mrr = evaluate()
    print('bert pretrain score is  hr1 = %.4f, hr3 = %.4f, hr5 = %.4f, hr10 = %.4f, hr20 = %.4f, ndcg1 = %.4f, ndcg3 = %.4f, ndcg5 = %.4f, ndcg10 = %.4f, ndcg20 = %.4f, map = %.4f, mrr = %.4f ' %
          (hr1, hr3, hr5, hr10, hr20, ndcg1, ndcg3, ndcg5, ndcg10, ndcg20, map, mrr))



    #JD-Xuetang bert pretrain score is  hr1 = 0.1257, hr3 = 0.2607, hr5 = 0.3476, hr10 = 0.4732, hr20 = 0.6082, ndcg1 = 0.1257, ndcg3 = 0.2034, ndcg5 = 0.2391, ndcg10 = 0.2793, ndcg20 = 0.3134, map = 0.2380, mrr = 0.2380 




