import pickle
import pandas as pd
'''
with open('args_data.pkl', 'rb') as f:
        item_key, brand_key, msort_key, user_key = pickle.load(f)
        brand_list = pickle.load(f)
        msort_list = pickle.load(f)
        user_count, item_count, brand_count, msort_count,example_count = pickle.load(f)
print('user_count: %d\titem_count: %d\tbrand_count: %d\tmsort_count: %d' %
          (user_count, item_count, brand_count, msort_count))
'''



PATH_TO_PRED='pred_skn.txt'
pred_data = pd.read_csv(PATH_TO_PRED, sep='\t',header=None)
pred_data.columns=['UId','skn1','skn2','skn3','skn4','skn5','skn6','skn7','skn8','skn9','skn10','skn11','skn12','skn13',
              'skn14','skn15','skn16','skn17','skn18','skn19','skn20']
print('pred len',len(pred_data))

#pred_data['UId'] = pred_data['UId'].map(lambda x: user_key[x])
PATH_TO_TRUE='./data/click_data_20180416.txt'
true_data = pd.read_csv(PATH_TO_TRUE, sep='\t',header=None)
true_data.columns=['UId','ItemId','Time']
true_user=true_data['UId'].unique().tolist()
print('true uid',len(true_user))



pred_user=pred_data['UId'].unique().tolist()
print('pred uid',len(pred_user))
#print('user_key',len(user_key))

same_user=set(true_user)&set(pred_user)
print('same_user',len(same_user))

same_user=list(same_user)
recall_rate=0
precise_rate=0

top_1=0
top_5=0
top_10=0
top_20=0

for i in (same_user):
    pred_skn=pred_data[pred_data.UId==i].iloc[0][1:].values
    
    true_skn=true_data[true_data.UId==i].ItemId.values
  
    same_skn=set(pred_skn)&set(true_skn)
   
    recall_rate+=len(same_skn)/len(true_skn)
    precise_rate+=len(same_skn)/len(pred_skn)
    if(pred_skn[0]==true_skn[0]):
        top_1+=1
    pre_5=pred_skn[:5]
    if(true_skn[0] in pre_5):
        top_5+=1
    pre_10=pred_skn[:10]
    if(true_skn[0] in pre_10):
        top_10+=1    
    pre_20=pred_skn[:20]
    if(true_skn[0] in pre_20):
        top_20+=1
recall_rate=recall_rate/len(same_user)
precise_rate=precise_rate/len(same_user)
top_1=top_1/len(same_user)
top_5=top_5/len(same_user)
top_10=top_10/len(same_user)
top_20=top_20/len(same_user)
print('recall_rate',recall_rate)
print('precise_rate',precise_rate)
print('top_1',top_1)
print('top_5',top_5)
print('top_10',top_10)
print('top_20',top_20)

    
 
