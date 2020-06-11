negative_num = 1
num_course = 1951
course_corpus = './bert_association/standard_data/corpus/course_phrase.txt'
job_corpus = './bert_association/standard_data/corpus/job_phrase.txt'
train_pos_index = './bert_association/standard_data/train_data/train_standard_positive_pairs_index.txt'
train_tsv = './bert_association/standard_data/train.tsv'
# valid_pos_index = './bert_association//standard_data/valid_data/valid_positive_index.txt' #单独训练bert
valid_pos_index = './bert_association/standard_data/valid_data/standard_valid.txt' # 将bert融入框架
valid_tsv = './bert_association/standard_data/valid.tsv'
test_pos_neg_index = './bert_association/standard_data/eval_data/test_pointwise_index.txt' # the first pair is positive instance, the last 99 pairs are negative instances 
test_tsv = './bert_association/standard_data/test.tsv'




data_dir = '/home/jeremyhao/revise_paper/small_data_voting/bert_association/standard_data/'
bert_config_file = '/home/jeremyhao/revise_paper/small_data_voting/bert_association/chinese_L-12_H-768_A-12/bert_config.json'
task_name = 'jeremy'
vocab_file = '/home/jeremyhao/revise_paper/small_data_voting/bert_association/chinese_L-12_H-768_A-12/vocab.txt'
output_dir = '/home/jeremyhao/revise_paper/small_data_voting/bert_association/output/'
init_checkpoint = '/home/jeremyhao/revise_paper/small_data_voting/bert_association/chinese_L-12_H-768_A-12/bert_model.ckpt'
do_lower_case = True
max_seq_length = 128
do_train = True
do_eval = False
do_predict = True
train_batch_size = 32
eval_batch_size = 8
predict_batch_size = 8
learning_rate = 2e-5
num_train_epochs = 3.0
warmup_proportion = 0.1
save_checkpoints_steps = 1000
iterations_per_loop = 1000
use_tpu = False
tpu_name = None
tpu_zone = None
gcp_project = None
master = None
num_tpu_cores = 8