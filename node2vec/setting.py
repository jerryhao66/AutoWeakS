class setting():
    def __init__(self):
        self.input = './node2vecalgo/data/all_adjlist.txt'
        self.output = './node2vecalgo/vector.txt'
        self.number_walks = 10
        self.walk_length = 80
        self.representation_size = 128
        self.epochs = 10
        self.graph_format = 'adjlist'
        self.workers = 20
        self.window_size = 10
        self.p = 0.25
        self.q = 0.25
        # self.job_list = './node2vecalgo/data/job_12_15.txt'
        # self.course_list = './node2vecalgo/data/p_n_course_12_15.txt'
        self.job_list = './node2vecalgo/testset/test_job.txt'
        self.course_list = './node2vecalgo/testset/test_course.txt'

        self.weight_file='./node2vecalgo/weight/all_tf_idf.txt'


        # job course similarity
        self.top_relevant_courses = 60
        self.unsupervised_similarity_file = './node2vecalgo/similarity/node2vec_eval.txt'
        self.job_index = './node2vecalgo/similarity/job_list.txt'
        self.course_index = './node2vecalgo/similarity/course_list.txt'
        # self.EVAL_PATH = './node2vecalgo/similarity/standard_eval.txt'
        self.EVAL_PATH = './node2vecalgo/evalset/standard_eval.txt'
        self.training_max_index = 566