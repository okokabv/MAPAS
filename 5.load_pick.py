import csv
import pandas as pd
import re
import string
import numpy as np
import time
import pickle
import gzip
import os


with gzip.open('./Grad_CAM_output_data/heatmap.pickle', 'rb') as f:  
    heatmap = pickle.load(f)

heatmap = heatmap.reset_index(drop=True)
heat_word =  heatmap['kw'].values
heat_score = heatmap['heat'].values

mal_scoreCNT=0
be_scoreCNT=0
zero_scoreCNT =0
for i in heat_score :
    #if i <= 0 :
    if i > 0 :
        be_scoreCNT +=1
    elif i< 0 :
        mal_scoreCNT +=1 
    elif float(i)==float(0):
        zero_scoreCNT +=1

#mal_heat_word = heat_word [0:scoreCNT]
be_word = heat_word[0:be_scoreCNT]
be_word = be_word.tolist()

mal_heat_word = heat_word [(be_scoreCNT+zero_scoreCNT):len(heat_score)]
mal_heat_word = mal_heat_word.tolist()


print("mal",len(mal_heat_word))
print("be",len(be_word))
print(len(heat_score))

with gzip.open('./output/mal_feature_acg.pickle', 'wb') as f:
    pickle.dump(mal_heat_word, f)
#    pickle.dump(be_word, f)

with gzip.open('./output/be_feature_acg.pickle', 'wb') as f:
    pickle.dump(be_word, f)
#    pickle.dump(mal_heat_word, f)
