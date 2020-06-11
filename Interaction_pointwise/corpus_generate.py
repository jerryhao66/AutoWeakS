from Interaction_pointwise import setting
def generate_corpus(corpus_path):
    corpus_list = []
    with open(corpus_path, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            per_row_keyword = []
            arr = line.strip().split(' ')
            for item in arr:
                per_row_keyword.append(int(item))
            corpus_list.append(per_row_keyword)

            line = f.readline()
    return corpus_list
def generate_corpus_tf_idf(corpus_path):
    corpus_list = []
    with open(corpus_path, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            per_row_keyword = []
            arr = line.strip().split(' ')
            for item in arr:
                per_row_keyword.append(float(item))
            corpus_list.append(per_row_keyword)

            line = f.readline()
    return corpus_list


if __name__ == '__main__':
    course_corpus_list = generate_corpus(setting.COURSE_CORPUS)
    job_courpus_list = generate_corpus(setting.JOB_COUPUS)

    print(len(course_corpus_list))
    print(len(job_courpus_list))
