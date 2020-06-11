from setting import setting
import numpy as np
import math
import heapq


def load_vector_to_dict(filename):
    line_vector_dict = {}
    with open(filename, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            line = line.strip()
            arr = line.split(' ')
            # print(arr)
            line_vector_dict[arr[0]] = arr[1:]
            line = f.readline()
    return line_vector_dict


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
            course_list.append(arr[0])
            line = f.readline()
    return course_list


def dot_product(c, j):  # course, job
    c = np.reshape(np.array(c), (-1))
    j = np.reshape(np.array(j), (-1))
    return np.dot(c, j)


def evaluate(vector, args):
    job_list = load_job_list(args.job_index)
    course_list = load_course_list(args.course_index)

    # print(job_list)
    # print(course_list)
    with open(args.unsupervised_similarity_file, 'w', encoding='utf-8') as writer:
        for i in range(len(job_list)):
            predictions = []
            candidate_courses_list = course_list
            job_ebd = np.array(vector[job_list[i]]).astype(np.float32)
            for item in candidate_courses_list:
                course_ebd = np.array(vector[item]).astype(np.float32)
                predictions.append(cos_sim(course_ebd, job_ebd))
            map_course_score = {candidate_courses_list[t]: predictions[t] for t in range(len(candidate_courses_list))}
            ranklist = heapq.nlargest(args.top_relevant_courses, map_course_score, key=map_course_score.get)
            # print(ranklist)
            writer.write(str(job_list[i]))
            writer.write('\t')
            for item in ranklist:
                writer.write(str(item))
                writer.write('\t')
            writer.write('\n')

if __name__ == '__main__':
    args = setting()
    vector = load_vector_to_dict('./vector.txt')
    evaluate(vector, args)
