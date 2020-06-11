from bert_serving.client import BertClient
import numpy as np
bc = BertClient()
job_result, course_result = [], []
value = 0.9
with open('job_phrase.txt', 'r', encoding='utf-8') as f:
	line = f.readline()
	while line!="" and line!=None:
		current_list = []
		arr = line.strip()
		current_list.append(arr)
		current_ebd = bc.encode(current_list)
		job_result.append(current_ebd[0])
		# print(np.array(current_ebd[0]).shape) 
		# current_ebd (1, 768)  current_ebd[0] (768,) 

		line = f.readline()

npy_job_result = np.array(job_result)
print(npy_job_result.shape) # (706, 708 )
np.save('bert_pretrain_job.npy', npy_job_result)


with open('course_phrase.txt', 'r', encoding='utf-8') as f:
	line = f.readline()
	while line!="" and line!=None:
		current_list = []
		arr = line.strip()
		current_list.append(arr)
		current_ebd = bc.encode(current_list)
		course_result.append(current_ebd[0])
		line = f.readline()

npy_course_result = np.array(course_result)
print(npy_course_result.shape)
np.save('bert_pretrain_course.npy', npy_course_result)

