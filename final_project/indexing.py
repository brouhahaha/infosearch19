import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
from math import log
import random
import csv
from gensim.models import Word2Vec, KeyedVectors
import re

with open('quora_question_pairs_rus.csv', 'r', encoding = 'utf-8') as f:  
    spamreader = csv.reader(f)
    all_data = [row[1:] for row in spamreader][1:]
    corpus = [row[1] for row in all_data]

k = 2.0
b = 0.75
lengths = [len(doc) for doc in corpus]
avgdl = sum(lengths)/len(lengths)
N = len(corpus)

filename = 'lemmatized.txt' #lemmatized.txt содержит преобработанный корпус quora_question_pairs - токенизированный, лемматизированный
filename_save_pickle_tf = 'tf.pickle'

def q_score(q, N, doc, len_doc, tf_matrix):        #подсчёт score отдельного слова
    n = np.count_nonzero(tf_matrix[q])
    tf = len(re.findall(' '+q+' ', doc))
    idf = IDF(N, n)
    score = idf*((tf*(k+1))/(tf+k*(1-b+b*len_doc/avgdl)))
    return score
    
def bm25(query, doc) -> float:   #query подать уже нормализованную))
    doc_contents = doc.split(' ')
    len_doc = len(doc_contents)
    scores = []
    for q in query:
        if q in vectorizer.get_feature_names():
            scores.append(q_score(q, N, doc, len_doc))
    return sum(scores)

def IDF(N, n):   # qi - одно слово из запроса
    idf = log((N-n+0.5)/(n+0.5))
    return idf

def get_preproc_docs(filename):                       
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        preprocessed_docs = content.split('\t')
    return preprocessed_docs

def create_tf_idf_matrix(preprocessed_docs):
    vectorizer = CountVectorizer(preprocessed_docs)
    X = vectorizer.fit_transform(preprocessed_docs[:10000])
    tf_matrix = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names())
    return tf_matrix

def create_bm25_matrix(tf_matrix, preprocessed_docs):
    bm25_matrix = pd.DataFrame(columns=list(tf_matrix.columns))
    for text in preprocessed_docs[:10000]:
        to_m_scores = []
        for item in list(tf_matrix.columns):
            if item in text.split():
                len_doc = len(text.split())
                to_m_scores.append(q_score(item, N, text, len_doc, tf_matrix))
            else:
                to_m_scores.append(0)
        bm25_matrix = bm25_matrix.append(pd.Series(to_m_scores, index=bm25_matrix.columns ), ignore_index=True)
    return bm25_matrix

def create_fasttext_matrix(model_fasttext, docs_lists_for_fasttext):
    fasttext_matrix = pd.DataFrame(columns = [i for i in range(300)])
    for document in docs_lists_for_fasttext[:20]:
        vecs_by_words = [model_fasttext[word] for word in document if word in model_fasttext.vocab]
        doc_vec = np.mean(vecs_by_words, axis = 0)
        fasttext_matrix = fasttext_matrix.append(pd.Series(doc_vec, index=fasttext_matrix.columns ), ignore_index=True)

def pickle_save(matrix, filename):
    pickle_out = open(filename,"wb")
    pickle.dump(matrix, pickle_out)
    pickle_out.close()

def csv_save(matrix, filename):
    matrix.to_csv(filename,index=False)


preproc_docs = get_preproc_docs('lemmatized.txt')
docs_lists_for_fasttext = [doc.split() for doc in preproc_docs]


#tf-idf
tf_matrix_new = create_tf_idf_matrix(preproc_docs)
pickle_save(tf_matrix_new, 'tf.pickle')

#bm25
bm25_matrix_new = create_bm25_matrix(tf_matrix_new, preproc_docs)
csv_save(bm25_matrix_new, 'bm25.csv')

#fasttext
model_fasttext = KeyedVectors.load("model.model")
fasttext_matrix = create_fasttext_matrix(model_fasttext, docs_lists_for_fasttext)
pickle_save(fasttext_matrix, 'ftext.pickle')


