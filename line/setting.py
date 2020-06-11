class setting():
    def __init__(self):
        #line
        self.input1 = './linealgo/data/course_edges.txt'
        self.input2 = './linealgo/data/job_edges.txt'
        self.input = './linealgo/data/all_adjlist.txt'
        self.output = './linealgo/vector.txt'
        self.order = 3
        self.number_walks = 10
        self.directed = True
        self.walk_length = 80
        self.representation_size = 128
        self.epochs = 5
        self.method = 'line'
        self.graph_format = 'adjlist'
        self.negative_ratio = 5
        self.lr = 0.001
        # self.job_list = './linealgo/data/job_12_15.txt'
        # self.course_list = './linealgo/data/p_n_course_12_15.txt'

        self.job_list = './linealgo/testset/test_job.txt'
        self.course_list = './linealgo/testset/test_course.txt'

        self.model_checkpoint_path = './linealgo/Checkpoint/'
        self.batch_size = 1000
        self.weight_file = './linealgo/weight/all_tf_idf.txt'

        # job course similarity 
        self.top_relevant_courses = 60
        self.unsupervised_similarity_file = './linealgo/similarity/line_eval.txt'
        self.job_index = './linealgo/similarity/job_list.txt'
        self.course_index = './linealgo/similarity/course_list.txt'
        # self.EVAL_PATH = './linealgo/similarity/standard_eval.txt'
        self.EVAL_PATH = './linealgo/evalset/standard_eval.txt'
        self.training_max_index = 566



