import numpy as np
import heapq
import math


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


# afternoon look at this
def read_pair_point():
    job_course_dict = {}
    with open('./similarity/bm25_point_sigmoid.txt', 'r') as f:
        line = f.readline()
        while line!="" and line!=None:
            arr = line.strip().split(' ')
            job, course, point = arr[0], arr[1], float(arr[2])
            if job not in job_course_dict:
                job_course_dict[job] = {}
                job_course_dict[job][course] = point
            else:
                job_course_dict[job][course] = point
            line = f.readline()
    return job_course_dict

def getMRR(ranklist, gtItem):
    for index, item in enumerate(ranklist):
        if item == gtItem:
            return 1.0 / (index + 1.0)
    return 0

def _init_sim():
    job_list = load_job_list('./similarity/job_list.txt')
    course_list = load_course_list('./similarity/course_list.txt')
    job_course_dict = {}
    for i in range(len(job_list)):
        predictions = []
        for _ in course_list:
            predictions.append(0.0)
        map_course_score = {course_list[t][1:]:predictions[t] for t in range(len(course_list))}
        job_course_dict[job_list[i][1:]] = map_course_score
    return job_course_dict

def load_eval_list(eval_path):
    with open(eval_path, 'r', encoding='utf-8') as f:
        job_list, course_list, label_list = [], [], []
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            job, course, label = arr[0], arr[1], arr[2]
            job_list.append(job)
            course_list.append(course)
            label_list.append(label)
            line = f.readline()
    return job_list, course_list, label_list

def cal_eval_reward():
    # cal eval reward bm25
    job_course_dict = read_pair_point()
    job_list, course_list, label_list = load_eval_list('./evalset/standard_eval.txt')
    job_list = np.array(job_list).reshape(-1, 100)
    course_list = np.array(course_list).reshape(-1, 100)
    eval_num_batch = len(job_list)

    mrrs = []
    for batch_index in range(eval_num_batch):
        predictions = []
        for item_index in range(len(job_list[batch_index])):
            job_index = str(job_list[batch_index][item_index])
            course_index = str(course_list[batch_index][item_index])
            predictions.append(job_course_dict[job_index][course_index])
        gtItem = course_list[batch_index][0]
        map_course_score = {course_list[batch_index][t]: predictions[t] for t in range(len(course_list[batch_index]))}
        ranklist100 = heapq.nlargest(100, map_course_score, key=map_course_score.get)
        mrr = getMRR(ranklist100, gtItem)
        mrrs.append(mrr)
    final_mrr = np.array(mrrs).mean()
    return final_mrr

def evaluate(job_course_dict):
    job_list = load_job_list('./testset/test_job.txt')
    course_list = load_course_list('./testset/test_course.txt')


    assert len(job_list) == len(course_list)

    hits5, ndcgs5, hits20, ndcgs20, hits10, ndcgs10, maps, mrrs = [], [], [], [], [], [], [], []
    for i in range(len(job_list)):
        predictions = []
        candidate_courses_list = course_list[i]
        print(candidate_courses_list)
        job_index = job_list[i]

        for item in candidate_courses_list:

            predictions.append(job_course_dict[job_index[1:]][item[1:]])
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

if __name__ == '__main__':
    # job_course_dict = read_pair_point()
    # final_hr5, final_ndcg5, final_hr20, final_ndcg20, final_hr10, final_ndcg10, final_map, final_mrr = evaluate(job_course_dict)
    # print(
    #     ' hr5 = %.4f, ndcg5 = %.4f, hr20 = %.4f, ndcg20 = %.4f, hr10 = %.4f, ndcg10 = %.4f, map = %.4f, mrr = %.4f' % (
    #         final_hr5, final_ndcg5, final_hr20,
    #         final_ndcg20, final_hr10, final_ndcg10, final_map, final_mrr))
    # hr5 = 0.1621, ndcg5 = 0.1513, hr20 = 0.3018, ndcg20 = 0.1891, hr10 = 0.2087, ndcg10 = 0.1659, map = 0.1734, mrr = 0.1734
    mrr = cal_eval_reward()
    print(mrr)