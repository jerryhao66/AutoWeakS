import numpy as np
from Representation_pointwise import ModelRPO
import tensorflow as tf

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    Rpoint_graph = tf.Graph()
    Rpoint_sess = tf.Session(graph=Rpoint_graph, config=config)
    
    with Rpoint_graph.as_default():
        with Rpoint_sess.as_default():
            Rpointmodel = ModelRPO.RPointModelNetwork()
            Rpoint_saver = tf.train.Saver()
            Rpoint_sess.run(tf.global_variables_initializer())
            
            # training, restore the model to calculate reward 
            ModelRPO.training(Rpointmodel, Rpoint_sess, Rpoint_saver)
            # mrr = ModelRPO.cal_mrr(Rpointmodel, Rpoint_sess, Rpoint_saver)
            # print(mrr)
            # ModelRPO.cal_reward(Rpointmodel, Rpoint_sess, Rpoint_saver)

        Rpoint_sess.close()
