class setting():
    def __init__(self):
        self.input = './deepwalkalgo/data/all_adjlist.txt'
        self.output = './deepwalkalgo/vector.txt'
        self.number_walks = 10
        self.walk_length = 80
        self.representation_size = 128
        self.graph_format = 'adjlist'
        # self.job_list = './deepwalkalgo/data/job_12_15.txt'
        # self.course_list = './deepwalkalgo/data/p_n_course_12_15.txt'
        self.job_list = './deepwalkalgo/testset/test_job.txt'
        self.course_list = './deepwalkalgo/testset/test_course.txt'
        self.workers = 20
        self.window_size = 10

        # job course similarity
        self.top_relevant_courses = 60
        self.unsupervised_similarity_file = './deepwalkalgo/similarity/deepwalk_eval.txt'
        self.job_index = './deepwalkalgo/similarity/job_list.txt'
        self.course_index = './deepwalkalgo/similarity/course_list.txt'

        # self.EVAL_PATH = './deepwalkalgo/similarity/standard_eval.txt'
        self.EVAL_PATH = './deepwalkalgo/evalset/standard_eval.txt'
        self.training_max_index = 566