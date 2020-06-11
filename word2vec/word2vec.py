from gensim.models import word2vec

import os
import gensim

import jieba
from gensim.models import word2vec
import numpy as np

#  去掉中英文状态下的逗号、句号
def clearSen(comment):
    comment = comment.strip(' ')
    comment = comment.replace('、','')
    comment = comment.replace('~','。')
    comment = comment.replace('～','')
    comment = comment.replace('{"error_message": "EMPTY SENTENCE"}','')
    comment = comment.replace('…','')
    comment = comment.replace('\r', '')
    comment = comment.replace('\t', ' ')
    comment = comment.replace('\f', ' ')
    comment = comment.replace('/', '')
    comment = comment.replace('、', ' ')
    comment = comment.replace('/', '')
    comment = comment.replace(' ', '')
    comment = comment.replace(' ', '')
    comment = comment.replace('_', '')
    comment = comment.replace('?', ' ')
    comment = comment.replace('？', ' ')
    comment = comment.replace('了', '')
    comment = comment.replace('➕', '')
    return comment

# 用jieba进行分词
comment = open('./big_corpus.txt').read()
comment = clearSen(comment)
# jieba.load_userdict('./user_dict/userdict_food.txt')
comment = ' '.join(jieba.cut(comment))

# 分完词后保存到新的txt中
fo = open("./afterSeg.txt","w")
fo.write(comment)
print("finished!")
fo.close()




# 用 word2vec 进行训练
sentences=word2vec.Text8Corpus('./afterSeg.txt')
# 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100
model=word2vec.Word2Vec(sentences,min_count=3, size=50, window=5, workers=1)
print('training word2vec...')
model.train(sentences, total_examples=2657,  epochs=5)
model.save('embedding.w2v')

model.wv.save_word2vec_format("model.bin", binary=False)

vectors = {}
with open('./model.bin', 'r') as f:
    line = f.readline()
    while line!="" and line!=None:
        arr = line.strip().split(' ')
        print(arr)
        vectors[arr[0]] = arr[1:]
        line = f.readline()



def save_embeddings(filename):
    fout = open(filename, 'w')
    # node_num = len(vectors.keys())
    for node, vec in vectors.items():
        fout.write("{} {}\n".format(node,
                                    ' '.join([str(x) for x in vec])))
    fout.close()

save_embeddings('output.txt')

vector_dict = {}
with open('./output.txt', 'r') as f:
    line = f.readline()
    line = f.readline()
    while line!="" and line!=None:
        arr = line.strip().split(' ')
        vector_dict[arr[0]] = np.asarray(arr[1:], np.float32)
        line = f.readline()


all_array = []
count = 0
with open('./afterSeg.txt', 'r') as f:
    line = f.readline()
    while line!="" and line!=None:
        current_array = np.zeros(50,)
        arr = line.strip().split(' ')
        for item in arr:
            if item not in vector_dict:
                count +=1
            else:
                # print('yes')
                # print(vector_dict[item])
                current_array = current_array + vector_dict[item]
                # print(current_array)
        current_array / len(item)
        all_array.append(current_array)
        line = f.readline()
print(count)
all_array = np.array(all_array)
print(np.array(all_array).shape)
job_ebd = all_array[0:2820]
course_ebd = all_array[2820:]
print(job_ebd.shape)
print(course_ebd.shape)
np.save('big_job_ebd', job_ebd)
np.save('big_course_ebd', course_ebd)