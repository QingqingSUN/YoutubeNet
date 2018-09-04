import pickle
import pandas as pd
import numpy as np
import time
def pre_columns_name(recall_num):
    columns=['UId']
    for i in range(recall_num):
        skn='skn'+str(i)
        columns+=[skn]
    return columns

start_time=time.time()

PATH_TO_PRED='/home/Cyan/YoutubeNet_pkl/pred_skn.txt'
pred_data = pd.read_csv(PATH_TO_PRED, sep='\t',header=None)

col=pre_columns_name(200)
print(col)
pred_data.columns=col
print('pred len',len(pred_data))


PATH_TO_Label='/home/Cyan/YoutubeNet_pkl/data/click_data_20180416.txt'
label_data = pd.read_csv(PATH_TO_Label, sep='\t',header=None)
label_data.columns=['UId','ItemId','Time']
label_user=label_data['UId'].unique().tolist()
print('label uid',len(label_user))
pred_user=pred_data['UId'].unique().tolist()
print('pred uid',len(pred_user))


overlap_user=set(label_user)&set(pred_user)
print('overlap_user',len(overlap_user))

overlap_user=list(overlap_user)
recall_num=[20,40,50,80,100,130,150,180,200]
hit_num=np.zeros(len(recall_num))
label_num=0
#recall_rate=np.zeros(len(recall_num))
precise_rate=np.zeros(len(recall_num))
#print(recall_rate)
for i in overlap_user:
    pred_skn=pred_data[pred_data.UId==i].iloc[0][1:].values

    label_skn=label_data[label_data.UId==i].ItemId.values
    label_skn=set(label_skn)
    label_num+=len(label_skn)
    k=0
    for x in recall_num:
        pre_=pred_skn[:x]
        hit_skn=set(pre_)&(label_skn)
        hit_num[k]+=len(hit_skn)
        precise_rate[k]+=len(hit_skn)/x
        k+=1
recall_rate=hit_num/label_num
precise_rate/=len(overlap_user)
print('recall rate',recall_rate)
print('precise rate',precise_rate)
print('Done time',time.time()-start_time)
