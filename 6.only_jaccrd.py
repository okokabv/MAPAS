import csv
import pandas as pd
import re
import string
import numpy as np
import time
import pickle
import gzip
import os

mal_folder='./data/mal/'
be_folder='./data/be/'

with gzip.open('./output/be_feature_acg.pickle', 'rb') as f:
    be_feature_acg = pickle.load(f)
with gzip.open('./output/mal_feature_acg.pickle', 'rb') as f:
    feature_acg = pickle.load(f)

with gzip.open('./output_acg.pickle', 'rb') as f:
    data = pickle.load(f)


def removeAllOccur(a, i):
    try:
        while True : a.remove(i)
    except ValueError:
        pass



X_name = data[0].values

X = data.drop(columns=[0,'class']) #no name
len_cnt =len(X.columns)
X = X.iloc[:, 0:len_cnt].values
X = X.tolist()



for i in range(0,len(X)):
    removeAllOccur(X[i],str(0))
    removeAllOccur(X[i],0)


be_target = os.listdir(be_folder)
be_score_be = np.zeros(len(be_target),dtype = np.float32)
be_score = np.zeros(len(be_target),dtype = np.float32)

for i in range(0, len(be_target)):
    tokenized_doc1 = be_feature_acg
    tokenized_doc2 = X[i]
    tokenized_doc4 = feature_acg

    
    intersection_be = set(tokenized_doc1).intersection(set(tokenized_doc2))
    intersection = set(tokenized_doc4).intersection(set(tokenized_doc2))
    

    be_score_be[i] =  len(intersection_be) / len(tokenized_doc1)    

    be_score[i] =  len(intersection) / len(tokenized_doc4)



mal_target = os.listdir(mal_folder)
mal_score = np.zeros(len(mal_target),dtype = np.float32)
mal_score_be = np.zeros(len(mal_target),dtype = np.float32)

for i in range(len(be_target), (len(be_target)+len(mal_target))):
    tokenized_doc1 = be_feature_acg
    tokenized_doc2 = X[i]
    tokenized_doc4 = feature_acg

    intersection_be = set(tokenized_doc1).intersection(set(tokenized_doc2))
    intersection = set(tokenized_doc4).intersection(set(tokenized_doc2))

    
    mal_score_be[i-len(be_target)] =  len(intersection_be) / len(tokenized_doc1)
    
    mal_score[i-len(be_target)] =  len(intersection) / len(tokenized_doc4)
   

result =pd.DataFrame({'be_score':be_score,'mal_score':mal_score})

result1 =pd.DataFrame({'be_score_be':be_score_be,'mal_score_be':mal_score_be})


print("be aver_be",np.mean(be_score_be))
print("mal aver_be",np.mean(mal_score_be))

print("be aver",np.mean(be_score))
print("mal aver",np.mean(mal_score))

with open('./output/jacaard_score.pickle', 'wb') as f:
    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

with open('./output/be_jacaard_score.pickle', 'wb') as f:
    pickle.dump(result1, f, pickle.HIGHEST_PROTOCOL)




