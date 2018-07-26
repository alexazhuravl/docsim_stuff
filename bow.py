from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import csv
from operator import itemgetter


def get_categories(df):
    corpus_category = [x.lower() for x in df['Category'].tolist() if isinstance(x, str)]
    return corpus_category


def get_subcategory(df):
    corpus_subcategory = [x.lower() for x in df['Subcategory'].tolist() if isinstance(x, str)]
    return corpus_subcategory


def get_bow(corpus):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    data = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return data


def get_data(source):
    for f in source:
        df = pd.read_excel('/home/az/Documents/work/document-similarity/data/' + f + '.xlsx')
        categories = get_categories(df)
        subcategories = get_subcategory(df)
        dict_cat = get_bow(categories)
        dict_subcat = get_bow(subcategories)
        df_cat = pd.DataFrame(dict_cat)
        df_subcat = pd.DataFrame(dict_subcat)
        df_cat.to_csv('BOW_category_%s' % f + ".csv", sep=",", quotechar='"', index=False,quoting=csv.QUOTE_ALL)
        df_subcat.to_csv('BOW_subcategory_%s' % f + ".csv", sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)

def get_data_csv(df):
    level2 = df_innitie['level2'].unique().tolist()
    level3 = df_innitie['level3 names'].unique().tolist()
    level2_innitie = [x.lower() for x in level2 if isinstance(x, str) and len(x) >= 2]
    level3_innitie = [x.lower() for x in level3 if isinstance(x, str) and len(x) >= 2]
    dict_l2 = get_bow(level2_innitie)
    dict_l3 = get_bow(level3_innitie)
    df_cat = pd.DataFrame(dict_l2)
    df_subcat = pd.DataFrame(dict_l3)
    df_cat.to_csv('BOW_innitie_l2.csv', sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)
    df_subcat.to_csv('BOW_innitie_l3.csv', sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)


if __name__ == '__main__':
    df_innitie = pd.read_csv('/home/az/Documents/work/document-similarity/data/innitie_cataloge_20180706.csv')

    files = ['manageathome', 'daily', 'unforgetable']
    get_data(files)
    get_data_csv(df_innitie)
