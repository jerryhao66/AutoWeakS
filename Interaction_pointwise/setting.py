###############    unsupervised method setting   #################
DEEPWALK_VEC_PATH = './Interaction_pointwise/vector/deep_vec.txt'
LINE_VEC_PATH = './Interaction_pointwise/vector/line_vec.txt'
METAPATH2VEC_PATH = './Interaction_pointwise/vector/metapath_vec.txt'
NODE2VEC_PATH = './Interaction_pointwise/vector/node2vec_vec.txt'
PTE_VEC_PATH = './Interaction_pointwise/vector/pte_vec.txt'

##############    Corpus setting        ######################
COURSE_CORPUS = './Interaction_pointwise/standard_data/corpus/standard_course_index.txt'
JOB_COUPUS = './Interaction_pointwise/standard_data/corpus/standard_job_index.txt'
JOB_TF_IDF_CORPUS = './Interaction_pointwise/standard_data/corpus/standard_job_tf_idf_index.txt'
NEGATIVE_PAIRS = './Interaction_pointwise/standard_data/pairs/standard_negative_pairs_index.txt'
POSITIVE_PAIRS = './Interaction_pointwise/standard_data/pairs/train_standard_positive_pairs_index.txt'
TRAININGSET_PATH = './Interaction_pointwise/standard_data/pairs/train_pointwise.txt'
# TRIPLE_PARIWISE_PATH = './standard_data/pairs/triple_pairwise.txt'
CPU_COUNT =1

# TEST_JOB_COURSE_PATH = './Interaction_pointwise/standard_data/pairs/test_pointwise.txt'
TEST_JOB_COURSE_PATH = './Interaction_pointwise/testset/test_pointwise.txt'

NUM_COURSE = 1951
NEGATIVE_NUM = 1

#####################  Model #####################
BATCH_SIZE = 128
LEARNING_RATE = 0.005
EMB_DIM = 640
EPOCH = 7
VERBOSE = 3
RANK_MODEL_CHECKPOINT_PATH = './Interaction_pointwise/Checkpoint/'
TEST_BATCH_SIZE = 100
SIM_BATCH_SIZE = 100
EVAL_BATCH_SIZE = 100
WEIGHT_SIZE = 50
LAMBDA_BILINEAR = 1e-6

job_index ='./Interaction_pointwise/similarity/job_list.txt'
course_index = './Interaction_pointwise/similarity/course_list.txt'
EVAL_PATH = './Interaction_pointwise/evalset/standard_eval.txt'
