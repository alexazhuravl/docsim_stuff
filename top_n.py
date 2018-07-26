import os
import csv
import pandas as pd

daily = ["simreport_daily_category", "simreport_daily_category_1",
         "simreport_daily_subcategory", "simreport_daily_subcategory_1"]

unforgettable = ["simreport_unforgettable_category", "simreport_unforgettable_category_1",
                 "simreport_unforgettable_subcategory", "simreport_unforgettable_subcategory_1"]

manageathome = ["simreport_manageathome_category", "simreport_manageathome_category_1",
                "simreport_manageathome_subcategory", "simreport_manageathome_subcategory_1"]

data = [daily, unforgettable, manageathome]
for f in daily:
    df = pd.read_csv('/home/az/Documents/work/document-similarity/' + f + '.csv')
    data = df.groupby('daily').head(10)
    data.to_csv('top10_%s' % f + ".csv", sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)


for c in unforgettable:
    df = pd.read_csv('/home/az/Documents/work/document-similarity/' + c + '.csv')
    data = df.groupby('unforgettable').head(10)
    data.to_csv('top10_%s' % c + ".csv", sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)

for k in manageathome:
    df = pd.read_csv('/home/az/Documents/work/document-similarity/' + k + '.csv')
    data = df.groupby('manageathome').head(10)
    data.to_csv('top10_%s' % k + ".csv", sep=",", quotechar='"', index=False, quoting=csv.QUOTE_ALL)




