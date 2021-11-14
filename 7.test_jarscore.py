import pickle
import gzip
import numpy as np
import os
import pandas as pd
import time
import os
import psutil
import gc
from apscheduler.schedulers.background import BackgroundScheduler
import sys


def _check_usage_of_cpu_and_memory(dir):
    tot = []
    pid = os.getpid()
    py  = psutil.Process(pid)
    cpu_usage   = os.popen("ps aux | grep " + str(pid) + " | grep -v grep | awk '{print $3}'").read()
    cpu_usage   = cpu_usage.replace("\n","")

    pid_cpu = psutil.cpu_percent(interval=3, percpu=True)

    percentage_cpu = float(sum(pid_cpu)/80)
    print("\nCPU : ",pid_cpu)
    print("\nCPU% :", percentage_cpu)

    memory_usage  = py.memory_full_info()[8] / 2.**20

    print(f"\nCurrent pss_memory MB   : {memory_usage: 9.5f} MB")

    del[[pid,py,cpu_usage,pid_cpu,percentage_cpu,memory_usage]]
    gc.collect()


def _check_usage_of_cpu_and_memory_log():
    pid = os.getpid()
    py  = psutil.Process(pid)
    cpu_usage   = os.popen("ps aux | grep " + str(pid) + " | grep -v grep | awk '{print $3}'").read()
    cpu_usage   = cpu_usage.replace("\n","")

    pid_cpu = psutil.cpu_percent(interval=0.1, percpu=True)

    percentage_cpu = float(sum(pid_cpu)/80)

    print("\nCPU : ",pid_cpu)
    print("\nCPU% :", percentage_cpu)

    memory_usage  = py.memory_full_info()[8] / 2.**20

    print(f"\nCurrent pss_memory MB   : {memory_usage: 9.5f} MB")

    del[[pid,py,cpu_usage,pid_cpu,percentage_cpu,memory_usage]]
    gc.collect()


a =_check_usage_of_cpu_and_memory('./2000_start.txt')

del[[a]]

scheduler = BackgroundScheduler()
#log ='./2000_log.txt' 
scheduler.add_job( _check_usage_of_cpu_and_memory_log, 'interval', seconds=0.1)
scheduler.start()

#sys.stdout = open(log,'a')

start = time.time()

with open('./output/be_jacaard_score.pickle','rb') as f:
    be_data = pickle.load(f)

with gzip.open('../test_acg.pickle','rb') as f:
    test_data = pickle.load(f)

#print("test len ", len(test_data))
with open('./output/jacaard_score.pickle','rb') as f:
    data = pickle.load(f)

with gzip.open('./output/be_feature_acg.pickle', 'rb') as f:
    be_feature_acg = pickle.load(f)

with gzip.open('./output/mal_feature_acg.pickle', 'rb') as f:
    feature_acg = pickle.load(f)

be_folder = './data/be/'
mal_folder = './data/mal/'

#totlen = len(test_data)
def removeAllOccur(a, i):
    try:
        while True : a.remove(i)
    except ValueError:
        pass


be_score_be = be_data['be_score_be'].values
mal_score_be = be_data['mal_score_be'].values

be_score = data['be_score'].values
mal_score = data['mal_score'].values


X = test_data.drop(columns=[0,'class']) #no name
len_cnt =len(X.columns)
X = X.iloc[:, 0:len_cnt].values
X = X.tolist()


for i in range(0,len(X)):
    removeAllOccur(X[i],str(0))
    removeAllOccur(X[i],0)

be_aver = np.mean(be_score)
mal_aver = np.mean(mal_score)

aver = (mal_aver+be_aver)/2

be_aver_be = np.mean(be_score_be)
mal_aver_be = np.mean(mal_score_be)

aver_be = (mal_aver_be+be_aver_be)/2
be_cnt = 0
mal_cnt = 0
tokenized_doc1 = be_feature_acg
tokenized_doc3 = feature_acg
be_target = os.listdir(be_folder)
for i in range(0,len(be_target)):

    intersection_be = set(tokenized_doc1).intersection(set(X[i]))
    intersection = set(tokenized_doc3).intersection(set(X[i]))
    
    be_score_be =  len(intersection_be) / len(tokenized_doc1)    
    be_score =  len(intersection) / len(tokenized_doc3)
    

    if float(be_score_be) < float(aver_be):
        if float(be_score) >= float(aver):
            be_cnt +=1


mal_target = os.listdir(mal_folder)
for i in range(len(be_target), (len(be_target)+len(mal_target))):

    intersection_be = set(tokenized_doc1).intersection(set(X[i]))
    intersection = set(tokenized_doc3).intersection(set(X[i]))

    
    mal_score_be =  len(intersection_be) / len(tokenized_doc1)
    mal_score =  len(intersection) / len(tokenized_doc3)
       

    if float(mal_score_be) >= float(aver_be):
        if float(mal_score) < float(aver):
            mal_cnt+=1

'''
be_accuray = (len(be_target)-be_cnt)/len(be_target)
mal_accuray = (len(mal_target)-mal_cnt)/len(mal_target)
print(be_accuray)

print(mal_accuray)

accu = ((len(be_target)-be_cnt)+ (len(mal_target)-mal_cnt)) / (len(be_target)+len(mal_target))
print("accu",accu)

print("time",time.time()-start)
'''

be_accuray = (be_cnt)/len(be_target) # FP
mal_accuray = (mal_cnt)/len(mal_target) # TP

TP = mal_accuray
FP = 1-be_accuray
TN = be_accuray
FN = 1-TP

#TTP = TP/len(mal_target)
#FFP = FP/len(be_target)
#FFN = FN/len(mal_target)
#TTN = TN/len(be_target)

print(be_accuray)
print(mal_accuray)

#res_folder = 'out'
#print('ACC : ',mal_accuray,file=open('./result/'+res_folder+'/out.txt','a'))

        #accu = ((len(be_target)-be_cnt)+ (len(mal_target)-mal_cnt)) / (len(be_target)+len(mal_target))
        #print("accu",accu,file=open('./result/'+res_folder+'/out.txt','a'))
end_time = time.time()-start
#print("end_time",end_time)
#print("predict time : ",end_time,file=open('./result/'+res_folder+'/out.txt','a'))
Pre = TP/(TP+FP)
Re = TP/(TP+FN)
Acc = (TP+TN) / (TP+FN+FP+TN)
F1 = 2 * ((Pre*Re)/(Pre+Re))
res_folder = 'out'
#print('TP : ',TP,file=open('./result/'+res_folder+'/out.txt','a'))
#print('FP : ',FP,file=open('./result/'+res_folder+'/out.txt','a'))
#print('FN : ',FN,file=open('./result/'+res_folder+'/out.txt','a'))3
#print('TN : ',TN,file=open('./result/'+res_folder+'/out.txt','a'))
#print('Precision : ',Pre,file=open('./result/'+res_folder+'/out.txt','a'))
#print('Recall : ',Re,file=open('./result/'+res_folder+'/out.txt','a'))
#print('Accuracy : ',Acc,file=open('./result/'+res_folder+'/out.txt','a'))
#print('F1-Score : ',F1,file=open('./result/'+res_folder+'/out.txt','a'))

