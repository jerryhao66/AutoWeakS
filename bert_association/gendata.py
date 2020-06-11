import numpy as np

# global negative_num
# global num_course
# global course_corpus
# global job_corpus
# global train_pos_index
# global train_tsv
# global valid_pos_index
# global valid_tsv

from bert_association import setting


# negative_num = 1
# num_course = 1951

# course_corpus = './standard_data/corpus/course_phrase.txt'
# job_corpus = './standard_data/corpus/job_phrase.txt'
# train_pos_index = './standard_data/train_data/train_standard_positive_pairs_index.txt'
# train_tsv = './standard_data/train.tsv'
# valid_pos_index = './standard_data/valid_data/valid_positive_index.txt'
# valid_tsv = './standard_data/valid.tsv'
# test_pos_neg_index = './standard_data/eval_data/test_pointwise_index.txt' # the first pair is positive instance, the last 99 pairs are negative instances 
# test_tsv = './standard_data/test.tsv'

# course_corpus = './bert_association/standard_data/corpus/course_phrase.txt'
# job_corpus = './bert_association/standard_data/corpus/job_phrase.txt'
# train_pos_index = './bert_association/standard_data/train_data/train_standard_positive_pairs_index.txt'
# train_tsv = './bert_association/standard_data/train.tsv'
# valid_pos_index = './bert_association/standard_data/valid_data/valid_positive_index.txt'
# valid_tsv = './bert_association/standard_data/valid.tsv'
# test_pos_neg_index = './bert_association/standard_data/eval_data/test_pointwise_index.txt' # the first pair is positive instance, the last 99 pairs are negative instances 
# test_tsv = './bert_association/standard_data/test.tsv'

def generate_corpus():
    '''
    generate job, course corpus
    '''
    job_list, course_list = [], []
    with open(setting.course_corpus, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            current_result = []
            arr = line.strip()
            current_result.append(arr)
            course_list.append(current_result)
            line = f.readline()
    # print(course_list)
    # print(len(course_list))
    # print(course_list[0])

    with open(setting.job_corpus, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            current_result = []
            arr = line.strip()
            current_result.append(arr)
            job_list.append(current_result)
            line = f.readline()
    # print(len(job_list))
    # print(job_list[0])
    return course_list, job_list


def generate_data(pos_index_file, output_file, course_list, job_list):
    '''
    generate the training set or the validation set
    generate positive  and start negative sampling
    '''

    with open(output_file, 'w') as writer:
        positive_dict = {}
        with open(pos_index_file, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(' ')
                # print(arr)
                job, positive_course = int(arr[0]), int(arr[1])
                if job not in positive_dict:
                    positive_dict[job] = []
                    positive_dict[job].append(positive_course)
                else:
                    positive_dict[job].append(positive_course)
                line = f.readline()

        with open(pos_index_file, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(' ')
                # print(arr)
                job, pos_course = int(arr[0]), int(arr[1])
                writer.write('1' + ',' + str(job_list[job][0]) + ',' + str(course_list[pos_course][0]) + '\n')

                for t in range(setting.negative_num):
                    neg_course = np.random.randint(setting.num_course)
                    while neg_course in positive_dict[job]:
                        neg_course = np.random.randint(setting.num_course)
                    writer.write('0' + ',' + str(job_list[job][0]) + ',' + str(course_list[neg_course][0]) + '\n')

                line = f.readline()


# calculate validation set mrr
def generate_valid_data(valid_set_file, output_file, course_list, job_list):
    with open(output_file, 'w') as writer:
        with open(valid_set_file, 'r') as f:
            line = f.readline()
            while line!="" and line!=None:
                arr = line.strip().split(' ')
                job, course, label = int(arr[0]), int(arr[1]), int(arr[2])
                writer.write(str(label)+','+str(job)+','+str(course)+'\n')
                line = f.readline()


def generate_test_data(pos_neg_file, output_file, course_list, job_list):
    count = 0
    with open(output_file, 'w') as writer:
        with open(pos_neg_file, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                count += 1
                arr = line.strip().split(' ')
                # print(arr)
                job, course = int(arr[0]), int(arr[1])
                if count % 100 == 1:
                    writer.write('1' + ',' + str(job_list[job][0]) + ',' + str(course_list[course][0]) + '\n')
                else:
                    writer.write('0' + ',' + str(job_list[job][0]) + ',' + str(course_list[course][0]) + '\n')
                line = f.readline()


# if __name__ == '__main__':
#     course_list, job_list = generate_corpus()
#     generate_data(train_pos_index, train_tsv, course_list, job_list)
#     generate_data(valid_pos_index, valid_tsv, course_list, job_list)
#     generate_test_data(test_pos_neg_index, test_tsv, course_list, job_list)










