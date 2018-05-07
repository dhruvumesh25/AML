
# coding: utf-8

# In[1]:


import gensim
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from model2 import *
from sklearn.utils import shuffle
np.random.seed(4)


# In[2]:


# Load Google's pre-trained Word2Vec model.
embedding_vec = gensim.models.KeyedVectors.load_word2vec_format('../pretrained-embedding/GoogleNews-vectors-negative300.bin', binary=True)


# In[3]:


embedding_size = (embedding_vec['king']).shape[0]
# print(embedding_size)


# In[4]:


def construct_data_set(file_name):
    """
    Function that reads a tsv file into the memory.
    """
    print("constructing data set for {0}".format(file_name))
    train_data_set = []
    test_data_set = []
    dev_data_set = []
    paraphrase_df = pd.read_csv(file_name, header=None, names=["id", "qid1", "qid2", "text_a", "text_b", "label"])
    paraphrase_df = shuffle(paraphrase_df)
    
    return paraphrase_df

paraphrase_df = construct_data_set('../data/Quora_question_pair_partition/questions.csv')


# In[5]:


# print (train_paraphrase_df)


# In[5]:


oov_vector = np.random.rand(embedding_size)
start_vector = np.random.rand(embedding_size)
stop_vector = np.random.rand(embedding_size)
def get_sentence_embedding(sentence,context_window):
    ret = []
    sentence = re.findall('[a-zA-Z0-9]+', sentence)
    sentence_embeddings = [start_vector]
    for i in range(len(sentence)):
        try:
            word_vec = embedding_vec[sentence[i]]
            sentence_embeddings.append(word_vec)
        except KeyError:
            sentence_embeddings.append(oov_vector)
    sentence_embeddings.append(stop_vector)
    
    ret = list([np.concatenate(sentence_embeddings[i:i+context_window]).ravel() for i in range(len(sentence_embeddings)-context_window+1)])
    return np.asarray(ret)


# In[7]:


# z = get_sentence_embedding(train_paraphrase_df.iloc[7]['text_a'], 3)
# print(train_paraphrase_df.iloc[7]['text_a'])
# print(type(z))
# print(z.shape)


# In[6]:


hidden_size_attend = 600
output_size_attend = 300
hidden_size_compare = 900
output_size_compare = 300
hidden_size_aggregate1 = 150
hidden_size_aggregate2 = 75
output_size_aggregate = 1

dropout_attend = 0.9
dropout_compare = 0.95
dropout_aggregate = 1

# attend_model = AttendForwardNet(embedding_size,hidden_size,output_size,dropout)
# compare_model = CompareForwardNet(embedding_size,hidden_size,output_size,dropout)
# aggregate_model = AggregateForwardNet(,hidden_size,hidden_size,)

def build_whole_model(input_size):
#     sentence_a = get_sentence_embedding(train_paraphrase_df.iloc[index]['text_a'], context_window)
#     sentence_b = get_sentence_embedding(train_paraphrase_df.iloc[index]['text_b'], context_window)
    sentence_a = tf.placeholder(tf.float32,name="s_a",shape=[None,None,input_size])
    sentence_b = tf.placeholder(tf.float32,name="s_b",shape=[None,None,input_size])
    y = tf.placeholder(tf.float32,name="label",shape=[None,1])
    
    with tf.variable_scope("Model", reuse=tf.AUTO_REUSE) as scope:
        attend_sentence_a = build_model1(sentence_a,hidden_size_attend,output_size_attend,dropout_attend,scope="attend")
        attend_sentence_b = build_model1(sentence_b,hidden_size_attend,output_size_attend,dropout_attend,"attend")
        
        # tf.Print(attend_sentence_a)
        e = tf.matmul(attend_sentence_a,attend_sentence_b,transpose_b=True)
        # tf.Print(e)
        alpha_wts = tf.nn.softmax(e,axis=2,name="alpha")
        # tf.Print(alpha_wts)
        beta_wts = tf.nn.softmax(e,axis=1,name="beta")
            
        alpha = tf.matmul(alpha_wts, sentence_b)
        beta = tf.matmul(tf.transpose(beta_wts,perm=[0,2,1]), sentence_a)
        
        inp_compare_a = tf.concat([sentence_a,alpha],axis=2)
        inp_compare_b = tf.concat([sentence_b,beta],axis=2)
        
        compare_sentence_a = build_model1(inp_compare_a,hidden_size_compare,output_size_compare,dropout_compare,scope="compare")
        compare_sentence_b = build_model1(inp_compare_b,hidden_size_compare,output_size_compare,dropout_compare,"compare")
        
        sum_sentence_a = tf.reduce_sum(compare_sentence_a,axis=1)
        sum_sentence_b = tf.reduce_sum(compare_sentence_b,axis=1)
        
        inp_aggregate = tf.concat([sum_sentence_a,sum_sentence_b],axis=1)
        aggregate = build_model2(inp_aggregate,hidden_size_aggregate1,hidden_size_aggregate2,output_size_aggregate,dropout_aggregate,scope="aggregate")
        
        pred = tf.sigmoid(aggregate)
        
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=aggregate, labels=y))
    return pred,cost


# In[8]:


tf.reset_default_graph()
predictions,cost = build_whole_model(900)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)
context_window = 3
batch_size = 1
num_x_train = 300000
display_step = 1
# Initializing the global variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    # sess.run(init)
    saver.restore(sess, "./model.ckpt")

#     sentence_a = np.expand_dims(get_sentence_embedding(train_paraphrase_df.iloc[1]['text_a'], context_window),axis=0)
#     sentence_b = np.expand_dims(get_sentence_embedding(train_paraphrase_df.iloc[1]['text_b'], context_window),axis=0)
#     print(sess.run([predictions,cost],feed_dict={"s_a:0":sentence_a,"s_b:0":sentence_b}))
    
    for epoch in range(10):
        avg_cost = 0
        for i in range(num_x_train):
            try:
                x_a = np.expand_dims(get_sentence_embedding(paraphrase_df.iloc[i]['text_a'], context_window),axis=0)
                x_b = np.expand_dims(get_sentence_embedding(paraphrase_df.iloc[i]['text_b'], context_window),axis=0)
                y = np.expand_dims(np.array(paraphrase_df.iloc[i]['label']),axis=0).reshape(1,1)
            
                _, pred, c = sess.run([optimizer,predictions,cost], 
                                feed_dict={
                                    "s_a:0": x_a, 
                                    "s_b:0": x_b, 
                                    "label:0": y
                                })
                avg_cost += c
            except:
                pass
            if (i+1)%10000==0:
                print(avg_cost/(i+1))
                saver.save(sess,'./model.ckpt')
        saver.save(sess,'./model.ckpt')
        num_correct = 0
        for i in range(num_x_train):
            try:
                x_a = np.expand_dims(get_sentence_embedding(paraphrase_df.iloc[i]['text_a'], context_window),axis=0)
                x_b = np.expand_dims(get_sentence_embedding(paraphrase_df.iloc[i]['text_b'], context_window),axis=0)
                y = np.expand_dims(np.array(paraphrase_df.iloc[i]['label']),axis=0).reshape(1,1)
            
            
                pred = sess.run([predictions], 
                                feed_dict={
                                    "s_a:0": x_a, 
                                    "s_b:0": x_b, 
                                })
                if abs(pred[0]-y[0])<0.5:
                    num_correct += 1
            except:
                pass
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost),"train_accuracy=",num_correct)

    print("Optimization Finished!")
