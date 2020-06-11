from parse import *
from query import QueryProcessor
import operator
import math
import heapq
import numpy as np

def chinese_course_dict(filename):
    course_index_chinese_dict = {}
    with open(filename, 'r',) as f:
        line = f.readline()
        while line != "" and line != None:
            line = line.strip()
            arr = line.split(',')
            name, index = arr[0], arr[1]
            course_index_chinese_dict[index] = name
            line = f.readline()
    return course_index_chinese_dict

def evaluate_ranking_output(test_filename, ranking_results):
    hits20, ndcgs20, hits10, ndcgs10, hits5, ndcgs5, maps, mrrs = [], [], [], [], [], [], [], []
    with open(test_filename, 'r') as f:
        line = f.readline()
        while line!="" and line!=None:
            line = line.strip()
            arr = line.split(' ')
            job_id = int(arr[0])
            candidate_courses_list = arr[1:]
            gtItem = candidate_courses_list[-1]
            map_course_score = {candidate_courses_list[t]: ranking_results[job_id][int(candidate_courses_list[t])] for t in range(len(candidate_courses_list))}
            ranklist5 = heapq.nlargest(5, map_course_score, key=map_course_score.get)
            ranklist20 = heapq.nlargest(20, map_course_score, key=map_course_score.get)
            ranklist10 = heapq.nlargest(10, map_course_score, key=map_course_score.get)
            ranklist100 = heapq.nlargest(100, map_course_score, key=map_course_score.get)
            hr20 = getHitRatio(ranklist20, gtItem)
            ndcg20 = getNDCG(ranklist20, gtItem)
            hr10 = getHitRatio(ranklist10, gtItem)
            ndcg10 = getNDCG(ranklist10, gtItem)
            hr5 = getHitRatio(ranklist5, gtItem)
            ndcg5 = getNDCG(ranklist5, gtItem)
            ap = getAP(ranklist100, gtItem)
            mrr = getMRR(ranklist100, gtItem)
            hits20.append(hr20)
            ndcgs20.append(ndcg20)
            hits10.append(hr10)
            ndcgs10.append(ndcg10)
            hits5.append(hr5)
            ndcgs5.append(ndcg5)
            maps.append(ap)
            mrrs.append(mrr)

            line = f.readline()

    final_hr20, final_ndcg20, final_hr10, final_ndcg10, final_hr5, final_ndcg5, final_map, final_mrr = np.array(
        hits20).mean(), np.array(
        ndcgs20).mean(), np.array(hits10).mean(), np.array(ndcgs10).mean(), np.array(hits5).mean(), np.array(ndcgs5).mean(), np.array(maps).mean(), np.array(
        mrrs).mean()
    return (final_hr20, final_ndcg20, final_hr10, final_ndcg10, final_hr5, final_ndcg5, final_map, final_mrr)


    # with open(test_filename, 'r') as f:
    #     line = f.readline()
    #     while line!="" and line!=None:
    #         line = line.strip()
    #         arr = line.split(' ')
    #         job, course = int(arr[0]), int(arr[1])
    #         print(ranking_results[job][course])
    #         line = f.readline()




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


def main():


    course_index_chinese_dict = chinese_course_dict('../text/course.csv')
    # print(course_index_chinese_dict)

    qp = QueryParser(filename='../text/job_phrase.txt')
    # qp = QueryParser(filename='../text/mini_job_phrase.txt')
    qp.parse()
    queries = qp.get_queries()
    print(len(queries))
    # print(queries)
    # print(len(queries))

    cp = CorpusParser(filename='../text/course_phrase.txt')
    cp.parse()
    corpus = cp.get_corpus()
    # print(corpus)
    proc = QueryProcessor(queries, corpus)
    # print(proc.index.index)
    # print(proc.dlt.table.keys())
    
    results = proc.run()
    # print(results[0][725])
    print(len(results))

    # #top 60=1
    # with open('bm25_weaksupervision.csv', 'w', encoding='utf-8') as writer:
    #     for result in results:
    #         sorted_x = sorted(result.items(), key=operator.itemgetter(1))
    #         sorted_x.reverse()
    #         for i in sorted_x[:60]:
    #             writer.write(str(i[0])+',')
    #         writer.write('\n')


    # evaluate
    hr_20, ndcg_20, hr_10, ndcg_10, hr_5, ndcg_5,  f_map, f_mrr = evaluate_ranking_output(test_filename='../text/job_course_99neg_1pos.txt', ranking_results=results)
    print(
        'hr5 = %.4f, ndcg5 = %.4f, hr20 = %.4f, ndcg20 = %.4f, hr10 = %.4f, ndcg10 = %.4f, map = %.4f, mrr = %.4f' % (
        hr_5, ndcg_5, hr_20, ndcg_20, hr_10, ndcg_10, f_map,
        f_mrr))

    #write_output
    # with open('bm25_output.csv', 'w', encoding='utf-8') as writer:
    #     for result in results:
    #         sorted_x = sorted(result.items(), key=operator.itemgetter(1))
    #         print(sorted_x)
    #         sorted_x.reverse()
    #         print(sorted_x)
    #         for i in sorted_x[:100]:
    #             writer.write(course_index_chinese_dict[str(i[0])]+',')
    #         writer.write('\n')


if __name__ == '__main__':
    main()