import random
import pickle
import numpy as np
import time
random.seed(1234)


start_time = time.time()

with open('remap.pkl', 'rb') as f:
    userclick_data = pickle.load(f)
    item_key, brand_key, msort_key, user_key = pickle.load(f)
    brand_list = pickle.load(f)
    msort_list = pickle.load(f)
    user_count, item_count, brand_count,msort_count, example_count = pickle.load(f)

    print('user_count: %d\titem_count: %d\tbrand_count: %d\texample_count: %d' %
      (user_count, item_count, brand_count, example_count))
    train_set = []
    test_set = []
    uid_num=0
    for UId, hist in userclick_data.groupby('UId'):
        uid_num+=1

        pos_list = hist['ItemId'].tolist()

        if len(pos_list)<3:
            continue
  
        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count-1)
            return neg

        neg_list = [gen_neg() for i in range(20*len(pos_list))]
        neg_list=np.array(neg_list)
        

        for i in range(1, len(pos_list)):
            index = np.random.randint(len(neg_list), size=20)
    
            hist = pos_list[:i]
            #if(len(hist)>20)
                #hist=hist[-20:]
            if i!= len(pos_list) :
             
                train_set.append((UId, hist, pos_list[i], list(neg_list[index])))
             
   
        if len(pos_list)>20:
            test_set.append((UId, pos_list[-20:]))
        else:
            test_set.append((UId, pos_list))

train_len=len(train_set)    
print(len(train_set))
train_set_1=train_set[:400000]
train_set_2=train_set[400000:800000]
train_set_3=train_set[800000:]

random.shuffle(train_set)
#random.shuffle(test_set)



with open('dataset.pkl', 'wb') as f:
    print('train')
    pickle.dump(train_set_1, f, pickle.HIGHEST_PROTOCOL)
    print('2')
    pickle.dump(train_set_2, f, pickle.HIGHEST_PROTOCOL)
    print('3')
    pickle.dump(train_set_3, f, pickle.HIGHEST_PROTOCOL)
    print('test')
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(brand_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(msort_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, brand_count,msort_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((item_key, brand_key, msort_key,user_key) , f, pickle.HIGHEST_PROTOCOL)
print(' DONE\tCost time: %.2f' %
                      (time.time() - start_time))
