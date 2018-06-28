import csv

from gensim.models.keyedvectors import KeyedVectors
from DocSim import DocSim
import pandas as pd
import re

"""
googlenews_model_path = './data/GoogleNews-vectors-negative300.bin'
stopwords_path = "./data/stopwords_en.txt"

model = KeyedVectors.load_word2vec_format(googlenews_model_path, binary=True)
with open(stopwords_path, 'r') as fh:
    stopwords = fh.read().split(",")
ds = DocSim(model, stopwords=stopwords)

"""

def score(src, trg):
    sim_scores = ds.calculate_similarity(src, trg)
    return sim_scores


def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)


def clean(data):
    return data.apply(lambda x: striphtml(str(x))).apply(lambda x: x.lower())


def get_data(d):
    data = df[df['innitie_category'] == d]
    values = data['short_description'].values.tolist()
    cleaned_data = [x for x in values if x != 'nan']
    return cleaned_data


def fill_matrix(matrix, list):
    for x in range(len(list)):
        arr = score(list[x], list)
        matrix[matrix.columns[x]] = arr

    last = list[-1]
    arr_add = score(last, list)
    last_index = len(list) - 1
    matrix[matrix.columns[last_index]] = arr_add
    return matrix


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import re
    import gensim
    import nltk
    from nltk.tokenize import word_tokenize

    """
    df = pd.read_csv('/home/az/Documents/work/document-similarity/data/data_new.csv',
                     names=['id', 'innitie_category', 'product', 'long_description',
                            'short_description'])
    df['short_description'] = clean(df['short_description'])
    df['long_description'] = clean(df['long_description'])
    df['product'] = clean(df['product'])
    df = df.replace('nan', np.nan).dropna()
    vals = df['short_description'].values.tolist()
    ids = df['id'].values.tolist()
    matrix_new = pd.DataFrame(columns=vals, index=vals)
    fill_matrix(matrix_new, vals)
    matrix_new['id'] = ids
    matrix_new.to_csv('score_results.csv', sep=",", quotechar='"', index=False,
                  quoting=csv.QUOTE_ALL)
    """

    df = pd.read_excel('/home/az/Documents/work/document-similarity/data/data_categories.xlsx')
    df['desc'] = clean(df['desc'])
    raw_docs = df['desc'].tolist()
    gen_docs = [word_tokenize(text) for text in raw_docs]
    dictionary = gensim.corpora.Dictionary(gen_docs)
    print("Number of words in dictionary:", len(dictionary))
    for i in range(len(dictionary)):
        print(i, dictionary[i])
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    tf_idf = gensim.models.TfidfModel(corpus)

    s = 0
    for i in corpus:
        s += len(i)

    sims = gensim.similarities.Similarity('/home/az/Documents/work/document-similarity/data/', tf_idf[corpus],
                                          num_features=len(dictionary))
    query_doc = gen_docs[0]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    print(query_doc)
    print(sims[query_doc_tf_idf])