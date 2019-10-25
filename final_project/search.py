import pandas as pd
import numpy as np
import csv
import pickle
import sklearn
import pymorphy2 as pm2 
pmm = pm2.MorphAnalyzer()

from gensim.models import Word2Vec, KeyedVectors

import nltk 
from nltk.text import Text 
nltk.download("stopwords") 
from nltk.corpus import stopwords 
russian_stopwords = stopwords.words("russian")

with open(r'C:\Users\Maria\Desktop\infosearch_f\quora_question_pairs_rus.csv', 'r', encoding = 'utf-8') as f:  
    spamreader = csv.reader(f)
    all_data = [row[1:] for row in spamreader][1:]
    corpus = [row[1] for row in all_data]

def get_pickle_matrix(filename):
    pickle_in = open(filename,"rb")
    matrix = pickle.load(pickle_in)
    return matrix

def get_bm25matrix(filename):
    bm25_mx = pd.read_csv(filename)
    return bm25_mx

def preprocess_query(raw_query):
    normalized = [pmm.normal_forms(x)[0] for x in raw_query.split() if x not in russian_stopwords]
    return normalized

def preprocess_query_fasttext(model_path, QUERY):
    model_fasttext = KeyedVectors.load(model_path)
    query  = preprocess_query(QUERY)
    query_words_vecs = [model_fasttext[word] for word in query if word in model_fasttext.vocab]
    QUERY_vec = np.mean(query_words_vecs, axis = 0)
    QUERY_vec = QUERY_vec.reshape(1, -1)
    return QUERY_vec

def query_to_vector(QUERY, matrix):
    vector = [0] * len(list(matrix.columns))
    for i, word in enumerate(list(matrix.columns)):
        if word in QUERY:
            vector[i] = 1
    return vector

def get_result(raw_query, matrix):
    query = preprocess_query(raw_query)
    vector = query_to_vector(query, matrix)
    result_array  = np.dot(matrix, vector)
    sorted_result_array  = sorted([(e,i) for i,e in enumerate(result_array)], reverse = True)
    search_result = [(all_data[item[1]][1], item[0]) for item in list(sorted_result_array[:10])]
    return search_result

def get_fasttext_res(matrix, query, model_path):
    vector = preprocess_query_fasttext(model_path, query)
    cosine_fasttext_scores = []
    for i in range(matrix.shape[0]):
        docvec = np.array(matrix.iloc[i]).reshape(1, -1)
        similarity = sklearn.metrics.pairwise.cosine_similarity(vector, docvec)
        cosine_fasttext_scores.append(similarity[0][0])

    cosine_fasttext_scores_sorted = sorted([(e,i) for i,e in enumerate(cosine_fasttext_scores)], reverse = True)
    
    results = [(all_data[item[1]][1], item[0], all_data[item[1]][2]) for item in list(cosine_fasttext_scores_sorted[:10])]
    return results

#pickle_in = open("tf.pickle","rb")
#matrix = pickle.load(pickle_in)
#query = 'рождественские каникулы'
#matrix = get_pickle_matrix('ftext.pickle')
#get_fasttext_res(matrix, query, r'C:\Users\Maria\Desktop\infosearch_f\model.model')
#matrix = get_bm25matrix('bm25.csv')
#get_result(query, matrix)
