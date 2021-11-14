import pandas as pd
import numpy as np
from collections import Counter
import operator
import pickle
import gzip
from keras.preprocessing.sequence import pad_sequences

with gzip.open('output_data/output.pickle', 'rb') as f:
    data = pickle.load(f)


df = data.reset_index(drop=True) 
X = df.drop(columns=[0,'class']) 
len_cnt =len(X.columns)
X = X.iloc[:, 0:len_cnt].values

cnt = []
for i in range(0,len(X)):
    all = np.append(all,X[i])
    cnt.append(len(X[i]))

max_len = int(max(cnt))

all =  list(set(all))
all = pd.DataFrame(all)
all= all.astype(str)
all = all.sort_values(by=[0])
all = all.iloc[:, 0].values
all = list(all)
all.remove('<built-in function all>')
all = pd.DataFrame(all)

with gzip.open('output_data/api_call_word_list.pickle', 'wb') as f:
    pickle.dump(all, f)

X = X.tolist()
def removeAllOccur(a, i):
    try:
        while True : a.remove(i)
    except ValueError:
        pass

for i in range(0,len(X)):
    #print("X_be : ",X[i])
    removeAllOccur(X[i],str(0))
    removeAllOccur(X[i],0)

np.median([len(k) for k in X]), len(X)
keyword_cnt = Counter([i for item in X for i in item])
keyword_clip = sorted(keyword_cnt.items(), key=operator.itemgetter(1))[:]
keyword_clip_dict = dict(keyword_clip)
keyword_dict = dict(zip(keyword_clip_dict.keys(), range(len(keyword_clip_dict))))
keyword_dict['Padding'] = len(keyword_dict)
keyword_dict['Uknown'] = len(keyword_dict)
keyword_rev_dict = dict([(v,k) for k, v in keyword_dict.items()])

max_seq = max_len

def encoding_and_padding(corp_list, dic, max_seq=max_seq):
    coding_seq = [ [dic.get(j, dic['Uknown']) for j in i]  for i in corp_list ]
    return(pad_sequences(coding_seq, maxlen=max_seq, padding='pre', truncating='pre',value=dic['Padding']))

train_x = encoding_and_padding(X, keyword_dict, max_seq=int(max_seq))
train_y = data['class']


with gzip.open('output_data/train_x.pickle', 'wb') as f:
    pickle.dump(train_x, f)

with gzip.open('output_data/train_y.pickle', 'wb') as f:
    pickle.dump(train_y, f)


with gzip.open('output_data/keyword_dict.pickle', 'wb') as f:
    pickle.dump(keyword_dict, f)

with gzip.open('output_data/keyword_rev_dict.pickle', 'wb') as f:
    pickle.dump(keyword_rev_dict, f)
print(train_x)
