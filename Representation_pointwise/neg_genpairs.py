import numpy as np
from Representation_pointwise import setting

def negative(read_positive_path, write_negative_path, num_course, negative_num):

    positive_dict = {}
    with open(read_positive_path, 'r') as f:
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


    candadiate_list = []
    for i in range(1951):
        candadiate_list.append(int(i))
    # print(candadiate_list)

    with open(write_negative_path, 'w', encoding='utf-8')as writer:
        for key in positive_dict.keys():
            for t in range(negative_num):
                j = np.random.randint(num_course)
                while j in positive_dict[key]:
                    j = np.random.randint(num_course)
                writer.write(str(key)+' '+str(j)+'\n')


def generate_trainingset(pos_path, neg_path, outfile_path):

    dict1 = {}
    with open(pos_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            # print(arr)
            job, course = arr[0], arr[1]
            if job not in dict1:
                dict1[job] = {}
                dict1[job]['positive'] = []
                dict1[job]['negative'] = []
                dict1[job]['positive'].append(course)
            else:
                dict1[job]['positive'].append(course)

            line = f.readline()
    with open(neg_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(' ')
            # print(arr)
            job, course = arr[0], arr[1]
            if job not in dict1:
                dict1[job] = {}
                dict1[job]['negative'] = []
                dict1[job]['negative'].append(course)
            else:
                dict1[job]['negative'].append(course)

            line = f.readline()
    # print(dict1)

    with open(outfile_path, 'w', encoding='utf-8') as writer:
        for key in dict1:
            for positive_item in dict1[key]['positive']:
                writer.write(str(key)+' '+str(positive_item)+' '+str(1)+'\n')
                for negative_item in dict1[key]['negative']:
                    writer.write(str(key)+' '+str(negative_item)+' '+str(0)+'\n')
                    # print(key, positive_item, negative_item)


if __name__ == '__main__':
    num_course = setting.NUM_COURSE 
    negative_num = setting.NEGATIVE_NUM
    negative('../standard_data/pairs/train_standard_positive_pairs_index.txt', '../standard_data/pairs/standard_negative_pairs_index.txt', num_course, negative_num)

    generate_trainingset('../standard_data/pairs/train_standard_positive_pairs_index.txt', '../standard_data/pairs/standard_negative_pairs_index.txt','../standard_data/pairs/triple_pairwise.txt')

    # netavite(setting.POSITIVE_PAIRS, setting.NEGATIVE_PAIRS, setting.NUM_COURSE, setting.NEGATIVE_NUM)
    # generate_pairwise(setting.POSITIVE_PAIRS, setting.NEGATIVE_PAIRS, setting.TRIPLE_PARIWISE_PATH)