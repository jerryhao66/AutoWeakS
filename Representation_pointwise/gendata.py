from Representation_pointwise import corpus_generate, setting, pretrain_vector
import numpy as np
from Representation_pointwise.neg_genpairs import negative, generate_trainingset
import multiprocessing

course_corpus = None
job_corpus = None
_num_batch = None
job_all = None
course_all = None
label_all = None
keywords_num = None
_test_num_batch = None
test_job_all = None
test_course_all = None
test_job_tf_idf_all = None


def generate_corpus(course_corpus_path, job_corpus_path, job_tf_idf_path):
    global course_corpus
    global job_corpus
    global job_tf_idf_corpus
    global keywords_num
    _, _, _, _, _, keywords_num = pretrain_vector.load_data(Deepwalk_path=setting.DEEPWALK_VEC_PATH,
                                                            Line_path=setting.LINE_VEC_PATH,
                                                            Metapath2vec_path=setting.METAPATH2VEC_PATH,
                                                            Node2vec_path=setting.NODE2VEC_PATH,
                                                            Pte_path=setting.PTE_VEC_PATH)
    course_corpus = corpus_generate.generate_corpus(course_corpus_path)
    job_corpus = corpus_generate.generate_corpus(job_corpus_path)
    job_tf_idf_corpus = corpus_generate.generate_corpus_tf_idf(job_tf_idf_path)
    # print('course number is %d, job number is %d, tf_idf job number is %d' % (len(course_corpus), len(job_corpus), len(job_tf_idf_corpus)))


def load_training_list(train_path):
    job, course, job_tf_idf, label = [], [], [], []
    with open(train_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            job.append(job_corpus[int(arr[0])])
            course.append(course_corpus[int(arr[1])])
            job_tf_idf.append(job_tf_idf_corpus[int(arr[0])])
            label.append(int(arr[2]))
            line = f.readline()
    num_instance = len(job)
    return job, course, job_tf_idf, label, num_instance


def batch_gen(batches, i):
    return [(batches[r])[i] for r in range(4)]


def evaluate_batch_gen(batches, i):
    return [(batches[r])[i] for r in range(3)]

def eval_reward_batch_gen(batches, i):
    return [(batches[r])[i] for r in range(4)]


def generate_batch():
    global _num_batch
    global job_all
    global course_all
    global label_all
    global job_tf_idf_all
    generate_corpus(setting.COURSE_CORPUS, setting.JOB_COUPUS, setting.JOB_TF_IDF_CORPUS)
    negative(setting.POSITIVE_PAIRS, setting.NEGATIVE_PAIRS, setting.NUM_COURSE, setting.NEGATIVE_NUM)
    generate_trainingset(setting.POSITIVE_PAIRS, setting.NEGATIVE_PAIRS, setting.TRAININGSET_PATH)
    job_all, course_all, job_tf_idf_all, label_all, iterations = load_training_list(setting.TRAININGSET_PATH)
    _num_batch = int(iterations / setting.BATCH_SIZE)
    # print('num batch is %d' %_num_batch)
    return _preprocess(_get_train_batch_fixed)


def _preprocess(get_train_batch_fixed):
    job_list, course_list, job_tf_idf_list, label_list = [], [], [], []
    cpu_count = setting.CPU_COUNT
    if cpu_count == 1:
        # for i in range(100):
        for i in range(_num_batch):
            job_batch, course_batch, job_tf_idf_batch, label_batch = get_train_batch_fixed(i)
            job_list.append(job_batch)
            course_list.append(course_batch)
            job_tf_idf_list.append(job_tf_idf_batch)
            label_list.append(label_batch)
    else:
        pool = multiprocessing.Pool(cpu_count)
        res = pool.map(get_train_batch_fixed, range(_num_batch))
        pool.close()
        pool.join()
        job_list = [r[0] for r in res]
        course_list = [r[1] for r in res]
        job_tf_idf_list = [r[2] for r in res]
        label_list = [r[3] for r in res]
    return job_list, course_list, job_tf_idf_list, label_list


def _get_train_batch_fixed(i):
    batch_job_list, batch_course_list, batch_job_tf_idf_list, batch_num_job_list, batch_num_course_list, batch_num_job_tf_idf_list, batch_label_list = [], [], [], [], [], [], []
    begin_index = i * setting.BATCH_SIZE
    end_index = begin_index + setting.BATCH_SIZE
    for index in range(begin_index, end_index):
        if len(job_all[index]) > setting.MAX_PADDING_Q:
            batch_job_list.append(job_all[index][:setting.MAX_PADDING_Q])
            batch_job_tf_idf_list.append(job_tf_idf_all[index][:setting.MAX_PADDING_Q])
            batch_num_job_list.append(len(job_all[index][:setting.MAX_PADDING_Q]))
            batch_num_job_tf_idf_list.append(len(job_tf_idf_all[index][:setting.MAX_PADDING_Q]))
        else:
            batch_job_list.append(job_all[index])
            batch_job_tf_idf_list.append(job_tf_idf_all[index])
            batch_num_job_list.append(len(job_all[index]))
            batch_num_job_tf_idf_list.append(len(job_tf_idf_all[index]))

        if len(course_all[index]) > setting.MAX_PADDING_D:
            batch_course_list.append(course_all[index][:setting.MAX_PADDING_D])
            batch_num_course_list.append(len(course_all[index][:setting.MAX_PADDING_D]))
        else:
            batch_course_list.append(course_all[index])
            batch_num_course_list.append(len(course_all[index]))

        # batch_job_list.append(job_all[index])
        # batch_course_list.append(course_all[index])
        # batch_job_tf_idf_list.append(job_tf_idf_all[index])
        # batch_num_job_list.append(len(job_all[index]))
        # batch_num_course_list.append(len(course_all[index]))
        # batch_num_job_tf_idf_list.append(len(job_tf_idf_all[index]))

        batch_label_list.append(label_all[index])
    # batch_job_input = _add_mask(keywords_num, batch_job_list, max(batch_num_job_list))
    # batch_pos_course_input = _add_mask(keywords_num, batch_pos_course_list, max(batch_num_pos_course_list))
    # batch_neg_course_input = _add_mask(keywords_num, batch_neg_course_list,  max(batch_num_neg_course_list))
    # batch_job_tf_idf_input = _add_mask(0, batch_job_tf_idf_list, max(batch_num_job_tf_idf_list))
    # print(max(batch_num_job_list))
    # print(max(batch_num_pos_course_list))
    # print(max(batch_num_neg_course_list))
    # print(max(batch_num_job_tf_idf_list))

    batch_job_input = _add_mask(keywords_num, batch_job_list, setting.MAX_PADDING_Q)
    batch_course_input = _add_mask(keywords_num, batch_course_list, setting.MAX_PADDING_D)
    batch_job_tf_idf_input = _add_mask(0, batch_job_tf_idf_list, setting.MAX_PADDING_Q)

    # batch_job_input = _add_mask(keywords_num, batch_job_list, max(max(batch_num_job_list), max(batch_num_pos_course_list), max(batch_num_neg_course_list)))
    # batch_pos_course_input = _add_mask(keywords_num, batch_pos_course_list, max(max(batch_num_job_list), max(batch_num_pos_course_list), max(batch_num_neg_course_list)))
    # batch_neg_course_input = _add_mask(keywords_num, batch_neg_course_list, max(max(batch_num_job_list), max(batch_num_neg_course_list), max(batch_num_pos_course_list)))
    # batch_job_tf_idf_input = _add_mask(0, batch_job_tf_idf_list, max(max(batch_num_job_list), max(batch_num_neg_course_list), max(batch_num_pos_course_list), max(batch_num_job_tf_idf_list)))

    return batch_job_input, batch_course_input, batch_job_tf_idf_input, batch_label_list


def _add_mask(feature_mask, features, num_max):
    # uniformalize the length of each batch
    for i in range(len(features)):
        features[i] = features[i] + [feature_mask] * (num_max + 1 - len(features[i]))
    return features


##########################        test   ################################################
def load_test_double_list(test_double_path):
    test_job, test_course, test_job_tf_idf = [], [], []
    with open(test_double_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            # print(arr)
            test_job.append(job_corpus[int(arr[0])])
            test_course.append(course_corpus[int(arr[1])])
            test_job_tf_idf.append(job_tf_idf_corpus[int(arr[0])])
            line = f.readline()
    num_double = len(test_job)
    return test_job, test_course, test_job_tf_idf, num_double


def generate_test_batch():
    global _test_num_batch
    global keywords_num
    global test_job_all
    global test_course_all
    global test_job_tf_idf_all
    _, _, _, _, _, keywords_num = pretrain_vector.load_data(Deepwalk_path=setting.DEEPWALK_VEC_PATH,
                                                            Line_path=setting.LINE_VEC_PATH,
                                                            Metapath2vec_path=setting.METAPATH2VEC_PATH,
                                                            Node2vec_path=setting.NODE2VEC_PATH,
                                                            Pte_path=setting.PTE_VEC_PATH)
    generate_corpus(setting.COURSE_CORPUS, setting.JOB_COUPUS, setting.JOB_TF_IDF_CORPUS)
    test_job_all, test_course_all, test_job_tf_idf_all, test_iterations = load_test_double_list(
        setting.TEST_JOB_COURSE_PATH)
    _test_num_batch = int(test_iterations / setting.TEST_BATCH_SIZE)
    # print('test num batch is %d' % _test_num_batch)
    return _test_preprocess(_get_test_batch_fixed)


def _test_preprocess(get_test_batch_fixed):
    job_list, course_list, job_tf_idf_list = [], [], []
    cpu_count = setting.CPU_COUNT
    if cpu_count == 1:
        for i in range(_test_num_batch):
            job_batch, course_batch, job_tf_idf_batch = get_test_batch_fixed(i)
            job_list.append(job_batch)
            course_list.append(course_batch)
            job_tf_idf_list.append(job_tf_idf_batch)
    else:
        pool = multiprocessing.Pool(cpu_count)
        res = pool.map(get_test_batch_fixed, range(_test_num_batch))
        pool.close()
        pool.join()
        job_list = [r[0] for r in res]
        course_list = [r[1] for r in res]
        job_tf_idf_list = [r[2] for r in res]
    return job_list, course_list, job_tf_idf_list


def _get_test_batch_fixed(i):
    test_batch_job_list, test_batch_course_list, test_batch_job_tf_idf_list, test_batch_num_job_list, test_batch_num_course_list, test_batch_num_job_tf_idf_list = [], [], [], [], [], []
    begin_index = i * setting.TEST_BATCH_SIZE
    end_index = begin_index + setting.TEST_BATCH_SIZE
    for index in range(begin_index, end_index):
        if len(test_job_all[index]) > setting.MAX_PADDING_Q:
            test_batch_job_list.append(test_job_all[index][:setting.MAX_PADDING_Q])
            test_batch_job_tf_idf_list.append(test_job_tf_idf_all[index][:setting.MAX_PADDING_Q])
            test_batch_num_job_list.append(len(test_job_all[index][:setting.MAX_PADDING_Q]))
            test_batch_num_job_tf_idf_list.append(len(test_job_tf_idf_all[index][:setting.MAX_PADDING_Q]))

        else:
            test_batch_job_list.append(test_job_all[index])
            test_batch_job_tf_idf_list.append(test_job_tf_idf_all[index])
            test_batch_num_job_list.append(len(test_job_all[index]))
            test_batch_num_job_tf_idf_list.append(len(test_job_tf_idf_all[index]))

        if len(test_course_all[index]) > setting.MAX_PADDING_D:
            test_batch_course_list.append(test_course_all[index][:setting.MAX_PADDING_D])
            test_batch_num_course_list.append(len(test_course_all[index][:setting.MAX_PADDING_D]))
        else:
            test_batch_course_list.append(test_course_all[index])
            test_batch_num_course_list.append(len(test_course_all[index]))

        # test_batch_course_list.append(test_course_all[index])
        # test_batch_job_tf_idf_list.append(test_job_tf_idf_all[index])
        # test_batch_num_job_list.append(len(test_job_all[index]))
        # test_batch_num_course_list.append(len(test_course_all[index]))
        # test_batch_num_job_tf_idf_list.append(len(test_job_tf_idf_all[index]))
    test_batch_job_input = _add_mask(keywords_num, test_batch_job_list, setting.MAX_PADDING_Q)
    test_batch_course_input = _add_mask(keywords_num, test_batch_course_list, setting.MAX_PADDING_D)
    test_batch_job_tf_idf_input = _add_mask(0, test_batch_job_tf_idf_list, setting.MAX_PADDING_Q)

    # test_batch_job_input = _add_mask(keywords_num, test_batch_job_list, max(test_batch_num_job_list))
    # test_batch_course_input = _add_mask(keywords_num, test_batch_course_list,max(test_batch_num_course_list))
    # test_batch_job_tf_idf_input = _add_mask(0, test_batch_job_tf_idf_list, max(test_batch_num_job_tf_idf_list))

    # test_batch_job_input = _add_mask(keywords_num, test_batch_job_list, max(max(test_batch_num_job_list), max(test_batch_num_course_list)))
    # test_batch_course_input = _add_mask(keywords_num, test_batch_course_list, max(max(test_batch_num_job_list),max(test_batch_num_course_list)))
    # test_batch_job_tf_idf_input = _add_mask(0, test_batch_job_tf_idf_list, max(max(test_batch_num_job_list),max(test_batch_num_course_list),max(test_batch_num_job_tf_idf_list)))

    return test_batch_job_input, test_batch_course_input, test_batch_job_tf_idf_input


################################# calulate similarity ###################################
def generate_eval_batch():
    global _eval_num_batch
    global keywords_num
    global eval_job_all
    global eval_course_all
    global eval_job_tf_idf_all
    global eval_label
    _, _, _, _, _, keywords_num = pretrain_vector.load_data(Deepwalk_path=setting.DEEPWALK_VEC_PATH,
                                                            Line_path=setting.LINE_VEC_PATH,
                                                            Metapath2vec_path=setting.METAPATH2VEC_PATH,
                                                            Node2vec_path=setting.NODE2VEC_PATH,
                                                            Pte_path=setting.PTE_VEC_PATH)

    generate_corpus(setting.COURSE_CORPUS, setting.JOB_COUPUS, setting.JOB_TF_IDF_CORPUS)
    eval_job_all, eval_course_all, eval_job_tf_idf_all, eval_label, eval_iterations = load_eval_list(setting.EVAL_PATH)
    _eval_num_batch = int(eval_iterations / setting.SIM_BATCH_SIZE)
    return _eval_preprocess(_get_eval_batch_fixed)


def _eval_preprocess(get_eval_batch_fixed):
    job_list, course_list, job_tf_idf_list, label_list = [], [], [], []
    cpu_count = setting.CPU_COUNT
    if cpu_count == 1:
        for i in range(_eval_num_batch):
            job_batch, course_batch, job_tf_idf_batch, label_batch = get_eval_batch_fixed(i)
            job_list.append(job_batch)
            course_list.append(course_batch)
            job_tf_idf_list.append(job_tf_idf_batch)
            label_list.append(label_batch)
    else:
        pool = multiprocessing.Pool(cpu_count)
        res = pool.map(get_eval_batch_fixed, range(_eval_num_batch))
        pool.close()
        pool.join()
        job_list = [r[0] for r in res]
        course_list = [r[1] for r in res]
        job_tf_idf_list = [r[2] for r in res]
        label_list = [r[3] for r in res]
    return job_list, course_list, job_tf_idf_list, label_list


def _get_eval_batch_fixed(i):
    eval_batch_job_list, eval_batch_course_list, eval_batch_job_tf_idf_list, eval_batch_num_job_list, eval_batch_num_course_list, eval_batch_num_job_tf_idf_list, eval_batch_label_list = [], [], [], [], [], [], []
    begin_index = i * setting.SIM_BATCH_SIZE
    end_index = begin_index + setting.SIM_BATCH_SIZE
    for index in range(begin_index, end_index):
        if len(eval_job_all[index]) > setting.MAX_PADDING_Q:
            eval_batch_job_list.append(eval_job_all[index][:setting.MAX_PADDING_Q])
            eval_batch_job_tf_idf_list.append(eval_job_tf_idf_all[index][:setting.MAX_PADDING_Q])
            # sim_batch_num_job_list.append(len(sim_job_all[index][:setting.MAX_PADDING_Q]))
            # sim_batch_num_job_tf_idf_list.append(len(sim_job_tf_idf_all[index][:setting.MAX_PADDING_Q]))
        else:
            eval_batch_job_list.append(eval_job_all[index])
            eval_batch_job_tf_idf_list.append(eval_job_tf_idf_all[index])
            # sim_batch_num_job_list.append(len(sim_job_all[index]))
            # sim_batch_num_job_tf_idf_list.append(len(sim_job_tf_idf_all[index]))

        if len(eval_course_all[index]) > setting.MAX_PADDING_D:
            eval_batch_course_list.append(eval_course_all[index][:setting.MAX_PADDING_D])
            # sim_batch_num_course_list.append(len(sim_course_all[index][:setting.MAX_PADDING_D]))
        else:
            eval_batch_course_list.append(eval_course_all[index])
            # sim_batch_num_course_list.append(len(sim_course_all[index]))
        eval_batch_label_list.append(eval_label[index])

    eval_batch_job_input = _add_mask(keywords_num, eval_batch_job_list, setting.MAX_PADDING_Q)
    eval_batch_course_input = _add_mask(keywords_num, eval_batch_course_list, setting.MAX_PADDING_D)
    eval_batch_job_tf_idf_input = _add_mask(0, eval_batch_job_tf_idf_list, setting.MAX_PADDING_Q)
    eval_batch_label_input = np.array(eval_batch_label_list).reshape(-1, 1)

    return eval_batch_job_input, eval_batch_course_input, eval_batch_job_tf_idf_input, eval_batch_label_input

def load_eval_list(eval_path):
    eval_job, eval_course, eval_job_tf_idf, eval_label  = [], [], [], []
    with open(eval_path, 'r', encoding='utf-8') as f:
        job_list, course_list = [], []
        line = f.readline()
        while line!="" and line!=None:
            arr = line.strip().split(' ')
            job, course, label = arr[0], arr[1], arr[2]
            job_list.append(job)
            course_list.append(course)
            eval_label.append(label)
            line = f.readline()

    assert len(job_list) == len(course_list) == len(eval_label)
    for index in range(len(job_list)):
        eval_job.append(job_corpus[int(job_list[index])])
        eval_course.append(course_corpus[int(course_list[index])])
        eval_job_tf_idf.append(job_tf_idf_corpus[int(job_list[index])])
    num = len(eval_job)
    return eval_job, eval_course, eval_job_tf_idf, eval_label, num


# def load_sim_list(sim_job_path, sim_course_path):
#     sim_job, sim_course, sim_job_tf_idf = [], [], []
#     with open(sim_job_path, 'r', encoding='utf-8') as f1:
#         job_list = []
#         line = f1.readline()
#         while line != "" and line != None:
#             arr = line.strip().split(' ')
#             job = arr[0]
#             job_list.append(job[1:])
#             line = f1.readline()
#     with open(sim_course_path, 'r', encoding='utf-8') as f2:
#         course_list = []
#         line = f2.readline()
#         while line != "" and line != None:
#             arr = line.strip().split(' ')
#             course = arr[0]
#             course_list.append(course[1:])
#             line = f2.readline()
#     for job_index in job_list:
#         for course_index in course_list:
#             sim_job.append(job_corpus[int(job_index)])
#             sim_course.append(course_corpus[int(course_index)])
#             sim_job_tf_idf.append(job_tf_idf_corpus[int(job_index)])
#     num_sim = len(sim_job)

#     return sim_job, sim_course, sim_job_tf_idf, num_sim

if __name__ == '__main__':
    d, e, f, g = generate_batch()
    # a, b, c = generate_test_batch()
    # print(np.array(a).shape)
    # print(np.array(b).shape)
    # print(np.array(c).shape)
    for i in range(10):
        # train
        # print(d[i])
        # print(g[i])
        print(np.array(d[i]).shape) #[128, 1208]
        print(np.array(e[i]).shape) #[128, 1208]
        print(np.array(f[i]).shape) #[128, 1208]
        print(np.array(g[i]).shape) #[128, ]

        # test
        # print(np.array(a[i]).shape) #[128, 1208]
        # print(np.array(b[i]).shape) #[128, 1208]
        # print(np.array(c[i]).shape) #[128, 1208]



