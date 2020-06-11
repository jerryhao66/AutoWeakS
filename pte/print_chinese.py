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

if __name__ == '__main__':

    course_index_chinese_dict = {}
    with open('./course.csv', 'r', encoding='gbk') as f:
        line = f.readline()
        while line != "" and line != None:
            line = line.strip()
            arr = line.split(',')
            name, index = arr[0], arr[1]
            course_index_chinese_dict[index] = name
            line = f.readline()



    item_dict = {}
    with open('./vector.txt', 'r', encoding='utf-8') as f:
        line = f.readline()
        while line!="" and line!=None:
            line = line.strip()
            arr = line.split(' ')
            item_dict[arr[0]] = arr[1:]
            line = f.readline()
    # print(item_dict)

    course_list = []
    with open('./recommendation/12_17_course_index.txt', 'r', encoding='utf-8') as f:
        line = f.readline()
        while line != "" and line != None:
            line = line.strip()
            arr = line.split('\t')
            course_list.append(arr[0])
            line = f.readline()
    # print(course_list)

    job_list = []
    with open('./recommendation/12_17_job_index.txt', 'r', encoding='utf-8') as f:
        line = f.readline()
        while line != "" and line != None:
            line = line.strip()
            arr = line.split('\t')
            job_list.append(arr[0])
            line = f.readline()
    # print(job_list)

    # '''
    # debug
    #  '''
    # predictions = []
    # predictions.append(cos_sim(np.array(item_dict['j242']).astype(np.float32), np.array(item_dict['c822']).astype(np.float32)))
    # predictions.append(
    #     cos_sim(np.array(item_dict['j242']).astype(np.float32), np.array(item_dict['c1190']).astype(np.float32)))
    # predictions.append(
    #     cos_sim(np.array(item_dict['j242']).astype(np.float32), np.array(item_dict['c1515']).astype(np.float32)))
    # predictions.append(
    #     cos_sim(np.array(item_dict['j242']).astype(np.float32), np.array(item_dict['c451']).astype(np.float32)))
    # predictions.append(
    #     cos_sim(np.array(item_dict['j242']).astype(np.float32), np.array(item_dict['c297']).astype(np.float32)))
    # print(predictions)





    '''
    evaluate
    '''
    with open('./recommendation/auto_phrase_line_chinese_output_new.csv', 'w') as writer:
        for job in job_list:
            job_ebd = np.array(item_dict[job]).astype(np.float32)
            predictions = []
            for course in course_list:
                course_ebd = np.array(item_dict[course]).astype(np.float32)
                predictions.append(cos_sim(course_ebd, job_ebd))
            map_course_score = {course_list[t]: predictions[t] for t in range(len(course_list))}
            ranklist60 = heapq.nlargest(60, map_course_score, key=map_course_score.get)
            for item in ranklist60:
                writer.write(str(course_index_chinese_dict[item])+',')
            writer.write('\n')
