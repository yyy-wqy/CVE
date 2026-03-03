# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 08:20:39 2025

@author: 18236
"""

import pickle
with open('dataset_tvd.pickle', 'rb') as file:
    dataset = pickle.load(file)


with open('dataset_tvd1.pickle', 'rb') as file:
    dataset1 = pickle.load(file)

with open('dataset_tvd2.pickle', 'rb') as file:
    dataset2 = pickle.load(file)
    
with open('dataset_tvd3.pickle', 'rb') as file:
    dataset3 = pickle.load(file)

with open('dataset_tvd4.pickle', 'rb') as file:
    dataset4 = pickle.load(file)

with open('dataset_tvd5.pickle', 'rb') as file:
    dataset5 = pickle.load(file)

d = []
d.extend(dataset1)
d.extend(dataset2)
d.extend(dataset3)
d.extend(dataset4)
d.extend(dataset5)

dd =  []
for i in d:
    if len(i) < 6:
        dd.append(i)
    
with open('dataset_tvd_3.12.pickle', 'wb') as file:
    pickle.dump(dataset, file) 