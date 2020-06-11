from Interaction_pointwise import setting
import numpy as np


def load_data(Deepwalk_path, Line_path, Metapath2vec_path, Node2vec_path, Pte_path):
    deepwalk_vec_j, deepwalk_vec_c, deepwalk_vec_k = read_vector(Deepwalk_path)
    line_vec_j, line_vec_c, line_vec_k = read_vector(Line_path)
    metapath_vec_j, metapath_vec_c, metapath_vec_k = read_vector(Metapath2vec_path)
    node2vec_vec_j, node2vec_vec_c, node2vec_vec_k = read_vector(Node2vec_path)
    pte_vec_j, pte_vec_c, pte_vec_k = read_vector(Pte_path)
    # print(deepwalk_vec_j.shape, line_vec_j.shape, metapath_vec_j.shape, node2vec_vec_j.shape, pte_vec_j.shape)
    # print(deepwalk_vec_c.shape, line_vec_c.shape, metapath_vec_c.shape, node2vec_vec_c.shape, pte_vec_c.shape)
    # print(deepwalk_vec_k.shape, line_vec_k.shape, metapath_vec_k.shape, node2vec_vec_k.shape, pte_vec_k.shape)
    vec_con_j = np.concatenate((deepwalk_vec_j, line_vec_j, metapath_vec_j, node2vec_vec_j, pte_vec_j), axis=1)
    vec_con_c = np.concatenate((deepwalk_vec_c, line_vec_c, metapath_vec_c, node2vec_vec_c, pte_vec_c), axis=1)
    vec_con_k = np.concatenate((deepwalk_vec_k, line_vec_k, metapath_vec_k, node2vec_vec_k, pte_vec_k), axis=1)
    # print(deepwalk_vec.shape, line_vec.shape, metapath_vec.shape, node2vec_vec.shape, pte_vec.shape)
    # print(vec_con.shape)
    # vec_avg = average_vector(deepwalk_vec, line_vec, metapath_vec,node2vec_vec, pte_vec)
    job_num = vec_con_j.shape[0]
    course_num = vec_con_c.shape[0]
    keywords_num = vec_con_k.shape[0]
    return vec_con_j, vec_con_c, vec_con_k, job_num, course_num, keywords_num


def read_vector(filepath):
    dict_keyword, dict_job, dict_course = {}, {}, {}
    job_vector, course_vector, keyword_vector = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')

            if arr[0][0] == 'k':
                dict_keyword[int(arr[0][1:])] = arr[1:]

            if arr[0][0] == 'j':
                dict_job[int(arr[0][1:])] = arr[1:]
            if arr[0][0] == 'c':
                dict_course[int(arr[0][1:])] = arr[1:]
            line = f.readline()
    list_keyword = sorted(dict_keyword)
    list_job = sorted(dict_job)
    list_course = sorted(dict_course)
    for index in list_keyword:
        current_vector = []
        for item in dict_keyword[index]:
            current_vector.append(float(item))
        keyword_vector.append(current_vector)
        # print(dict_keyword[item])

    for index in list_course:
        current_vector = []
        for item in dict_course[index]:
            current_vector.append(float(item))
        course_vector.append(current_vector)

    for index in list_job:
        current_vector = []
        for item in dict_job[index]:
            current_vector.append(float(item))
        job_vector.append(current_vector)

    # print(np.array(keyword_vector).shape)
    # print(np.array(course_vector).shape)
    # print(np.array(job_vector).shape)

    return np.array(job_vector), np.array(course_vector), np.array(keyword_vector)


def average_vector(vec1, vec2, vec3, vec4, vec5):
    return (vec1 + vec2 + vec3 + vec4 + vec5) / 5


if __name__ == '__main__':
    vec_job, vec_course, vec_keyword, job_num, course_num, keywords_num = load_data(
        Deepwalk_path=setting.DEEPWALK_VEC_PATH, Line_path=setting.LINE_VEC_PATH,
        Metapath2vec_path=setting.METAPATH2VEC_PATH, Node2vec_path=setting.NODE2VEC_PATH, Pte_path=setting.PTE_VEC_PATH)

    print(vec_job.shape, vec_course.shape, vec_keyword.shape)
    print(job_num, course_num, keywords_num)