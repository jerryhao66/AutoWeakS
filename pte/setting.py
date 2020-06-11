class setting():
    def __init__(self):
        self.input1 = './ptealgo/data/course_adjlist.txt'
        self.input2 = './ptealgo/data/job_adjlist.txt'
        self.input = './ptealgo/data/all_adjlist.txt'
        self.output = './ptealgo/vector.txt'
        self.number_walks = 10
        self.directed = True
        self.walk_length = 80
        self.representation_size = 128
        self.epochs = 6
        self.method = 'line'
        self.graph_format = 'adjlist'
        self.negative_ratio = 100
        self.lr = 0.001
        # self.job_list = './ptealgo/data/job_12_15.txt'
        # self.course_list = './ptealgo/data/p_n_course_12_15.txt'
        self.job_list = './ptealgo/testset/test_job.txt'
        self.course_list = './ptealgo/testset/test_course.txt'

        self.model_checkpoint_path = './ptealgo/Checkpoint/'
        self.weight_file = './ptealgo/weight/all_tf_idf.txt'
        self.weight_file1 = './ptealgo/weight/course_tf_idf.txt'
        self.weight_file2 = './ptealgo/weight/job_tf_idf.txt'
        self.batch_size = 128

        # job course similarity
        self.top_relevant_courses = 150
        self.unsupervised_similarity_file = './ptealgo/similarity/pte_eval.txt'
        self.job_index = './ptealgo/similarity/job_list.txt'
        self.course_index = './ptealgo/similarity/course_list.txt'

        # self.EVAL_PATH = './ptealgo/similarity/standard_eval.txt'
        self.EVAL_PATH = './ptealgo/evalset/standard_eval.txt'
        self.training_max_index = 566
