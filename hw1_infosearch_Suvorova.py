#!/usr/bin/env python
# coding: utf-8

import os
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import nltk 
from nltk.text import Text 
import pymorphy2 as pm2 
pmm = pm2.MorphAnalyzer() 

nltk.download("stopwords") 
#--------# 
from nltk.corpus import stopwords 
russian_stopwords = stopwords.words("russian")

def texts(folder):
    for root, dirs, files in os.walk(folder):
        for name in files:
            yield (os.path.join(root, name))
            
def cleanText(text):
    text = text.lower()
    text = re.sub('-', ' ', text)
    text = re.sub(r'[^\w\s]','',text) 
    text = re.sub(r'\d', '', text) 
    text = re.sub(r'[A-Za-z]', '', text)
    text = [pmm.normal_forms(x)[0] for x in text.split() if x not in russian_stopwords] 
    for i in text:
        if i in russian_stopwords:
            text.remove(i)
    return ' '.join(text)

def get_norms(texts):
    for text in texts:
        with open(text, 'r', encoding = 'utf-8') as f:  
            cont = f.read() 
            normal = cleanText(cont)
            yield normal

def prepare_texts():
    texts_list = list(texts('friends'))
    norms = get_norms(texts_list)
    texts_norm = list(norms)
    corpus = texts_norm
    df = pd.DataFrame(data=corpus, columns=['texts'])
    return df

def get_top_n_words(reversing, n, corpus):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=reversing)
    return words_freq[:n]

def inverted_ind(term_doc):
    inverted = {}
    for (columnName, columnData) in term_doc.iteritems():
        entries_epizodes = columnData.to_numpy().nonzero()
        inverted[columnName] = list(entries_epizodes[0])
    return inverted


def most_popular_season_per_person(name_in_vocab, result):
    chandler = {}
    chandler['season1'] = sum(result[name_in_vocab].iloc[:20])
    chandler['season2'] = sum(result[name_in_vocab].iloc[21:45])
    chandler['season3'] = sum(result[name_in_vocab].iloc[46:66])
    chandler['season4'] = sum(result[name_in_vocab].iloc[67:92])
    chandler['season5'] = sum(result[name_in_vocab].iloc[93:116])
    chandler['season6'] = sum(result[name_in_vocab].iloc[117:141])
    chandler['season7'] = sum(result[name_in_vocab].iloc[142:165])
    return max(zip(chandler.values(), chandler.keys()))


def find_relevant_episodes(x, inverted, to_search):
    search_results = {}
    for token in to_search:
        norm_f = pmm.normal_forms(token)[0]
        try:
            search_results[token] = inverted[norm_f]
        except:
            print('одного из ваших слов не встретилось в коллекции :(')
    episodes_to_range = search_results.values()
    episodes_result = set.intersection(*[set(ep) for ep in episodes_to_range])
    return episodes_result

def range_episodes(relevant_episodes, df, to_search):
    ratings = {}
    for ep in list(relevant_episodes):
        rating = 0 
        text = df['texts'][ep].split()
        for i in range(len(to_search)):
            if text.index(pmm.normal_forms(to_search[i])[0]) - text.index(pmm.normal_forms(to_search[i-1])[0]) < 3:
                rating += 1
        ratings[ep] = rating   
    return ratings

def find_best(ratings):
    itemMaxValue = max(ratings.items(), key=lambda x: x[1])
    listOfKeys = list()
    for key, value in ratings.items():
        if value == itemMaxValue[1]:
            listOfKeys.append(key)
    return listOfKeys
 

def main():
    print('processing texts...')
    df = prepare_texts()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['texts'].values)
    result = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names())
    print('50 самых частотных слов', get_top_n_words(True, 50, df['texts']), '\n')
    print('50 самых нечастотных слов', get_top_n_words(False, 50, df['texts']), '\n')
    inverted = inverted_ind(result)
    print('Слова, присутствующие во всех док-тах')
    for i in inverted:
        if len(inverted[i])==165:
            print(i)
    print('\n', 'Самый популярный сезон у Чендлера', most_popular_season_per_person('чендлера', result))
    print('Самый популярный сезон у Моники', most_popular_season_per_person('моника', result), '\n')
    x = input('search (type here):')
    to_search = x.split()
    relevant_episodes = find_relevant_episodes(x, inverted, to_search)
    rating = range_episodes(relevant_episodes, df, to_search)
    best = find_best(rating)
    print('Вашему запросу соответствуют эпизоды: ', best)


main()



