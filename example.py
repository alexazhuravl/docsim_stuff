import csv
import json
import operator
from collections import OrderedDict, defaultdict

from gensim.models.keyedvectors import KeyedVectors
from pandas.io.json import json_normalize

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


def score(query_doc, gen_docs):
    dictionary = gensim.corpora.Dictionary(gen_docs)

    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    tf_idf = gensim.models.TfidfModel(corpus)

    sims = gensim.similarities.Similarity('/home/az/Documents/work/document-similarity/data/', tf_idf[corpus],
                                          num_features=len(dictionary))
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    return sims[query_doc_tf_idf].tolist()


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


def create_dict(dict, items, subcat_dict):
    stop = len(dict.items())
    start = stop - items
    data = []

    for key, value in dict.items():
        for i in range(start, stop):
            vals = {'source_id': key, 'subcategory': subcat_dict[key].lower(),
                    'data': {'target_id': sorted(value.items(), key=lambda x: x[1])[i][0],
                             'score': sorted(value.items(), key=lambda x: x[1])[i][1],
                             'target_category': subcat_dict[sorted(value.items(),
                                                                   key=lambda x: x[1])[i][0]].lower()}}
            data += [vals]
    return data


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import re
    import gensim
    from nltk.corpus import stopwords
    import spacy

    #comparing data using spacy

    stop_words = set(stopwords.words('english'))
    df_innitie = pd.read_csv('/home/az/Documents/work/document-similarity/data/innitie_cataloge_20180706.csv')
    level2 = df_innitie['level2'].unique().tolist()
    level2_innitie = [x.lower() for x in level2 if isinstance(x, str) and len(x) >= 2]
    df_unf = pd.read_excel('/home/az/Documents/work/document-similarity/data/unforgetable.xlsx')
    data_unf = df_unf['Subcategory'].unique().tolist()
    data_unf = [x.lower() for x in data_unf if isinstance(x, str) and len(x) >= 2]
    nlp = spacy.load('en_core_web_lg')

    tokens1 = [nlp(x) for x in level2_innitie]
    tokens2 = [nlp(x) for x in data_unf]

    for t in tokens1:
        for x in tokens2:
            data = [t.text, x.text, str(t.similarity(x))]





    """
    # comparison and saving to *.csv
    df = pd.read_csv('/home/az/Documents/work/document-similarity/data/innitie_cataloge_20180706.csv',
                     names=['level1', 'level2', 'level3', 'level4',
                            'names'])
    df['names'] = clean(df['names'])
    df['level2'] = clean(df['level2'])
    df['level3'] = clean(df['level3'])
    df
    df = df.replace('nan', np.nan).dropna()
    df
    # vals = df['short_description'].values.tolist()
    # ids = df['id'].values.tolist()
    # matrix_new = pd.DataFrame(columns=vals, index=vals)
    # fill_matrix(matrix_new, vals)
    # matrix_new['id'] = ids
    # matrix_new.to_csv('score_results.csv', sep=",", quotechar='"', index=False,
    #                   quoting=csv.QUOTE_ALL)
    #

    """




    """
    #top 10 indexes
    df = pd.read_excel('/home/az/Documents/work/document-similarity/data/data_new.xlsx')
    df['desc'] = clean(df['desc'])
    raw_docs = df['desc'].tolist()
    gen_docs = [word_tokenize(text) for text in raw_docs]

    for i in range(len(gen_docs)):
        gen_docs[i] = [word for word in gen_docs[i] if word.isalpha()]
        gen_docs[i] = [w for w in gen_docs[i] if not w in stop_words]
        for k in range(len(gen_docs[i])):
            gen_docs[i][k] = wl.lemmatize(gen_docs[i][k])

    matrix_new = pd.DataFrame(columns=raw_docs, index=raw_docs)
    fill_matrix(matrix_new, gen_docs)

    ids = df['id'].values.tolist()
    matrix_new.columns = ids
    matrix_new = matrix_new.rename(columns={x: y for x, y in zip(df.columns, range(0, len(df.columns)))})

    idx = 0
    matrix_new.insert(loc=idx, column='id', value=df['id'].values.tolist())
    sub_dict = dict(zip(df['id'], df['subcategory']))

    dict_ = matrix_new.set_index('id').T.to_dict('id')

    d = create_dict(dict_, 10, sub_dict)

    for i in range(len(d)):
        d[i]['count'] = 0
        if d[i]['data']['target_category'] == d[i]['subcategory']:
            d[i]['count'] += 1

    final_data = json.dumps(d)
    res = json_normalize(d)
    res = res.loc[res['count'] == 1]
    cnt = res.groupby('source_id').describe()
    res.to_csv('res_cleaned.csv')
    res = res.index.to_series().groupby(res['source_id']).first().reset_index(name='index')
    indexes = res['index'].values.tolist()
    iids = res['source_id'].values.tolist()
    formatted = [x % 10 if x >= 10 else x for x in indexes]
    newdict = dict(zip(iids, formatted))
    index_scores = json.dumps(newdict)
    res.to_csv('index_cleaned.csv')
    """

    """"
    #describe results
    cnt.to_csv('count_categories.csv')

    mean = gb['data.score'].agg(np.mean)
    test = res.groupby('source_id').describe()

    test.to_csv('describe_top_10.csv')

    data_new = create_dict(dict_, 35, sub_dict)
    res_new = json_normalize(data_new)
    gb2 = res_new.groupby('source_id')
    mean_new = gb2['data.score'].agg(np.mean)
    mean_new.to_csv('mean_overall.csv')
    test2 = res.groupby('source_id').describe()
    test2.to_csv('describe_overall.csv')
    matrix_new.to_csv('test_1106.csv', sep=",", quotechar='"', index=False,
                     quoting=csv.QUOTE_ALL)
    """
