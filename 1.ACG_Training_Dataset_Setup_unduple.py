import os
import csv
import pickle
import pandas as pd
import gzip
folder='data/'

target = os.listdir(folder)
cnt = 0
d = []
cls =[]
fname=[]
for k in range(0,len(target)):
    payload = folder + target[k]

    for root,dirs, files in os.walk(payload):
        for file in files:
            d.append([])
            #d[cnt].append(file)
            fname=file
            with open(payload + '/' + file, 'r',encoding='UTF-8') as reader:
                data = reader.read()
                lines = data.strip().split('\n')
                #lines.sort()

                for row in lines[1:]:
                    api = row.split(',')[0]
                    if api != 'Class':
                        d[cnt].append(api)

            if target[k] == 'mal':
                cls.append(1)
            elif target[k] == 'be':
                cls.append(0)
            d[cnt] = list(set(d[cnt]))
            d[cnt].insert(0, fname)

            cnt += 1
all = pd.DataFrame(d)
all = all.fillna(0)
all['class'] = cls

cnt =0
with gzip.open('output_data/output.pickle', 'wb') as f:
    pickle.dump(all, f)
print(all)

