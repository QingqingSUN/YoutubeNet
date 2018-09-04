import os,time,sys,datetime
d = datetime.datetime.now()
date_diff = 1
def gain_preN_day(d, n):

    days_cnt = datetime.timedelta(days = n)


    # print(datetime.timedelta(days = n))
    day_from = d - days_cnt
    date_from = datetime.datetime(day_from.year, day_from.month, day_from.day, 0, 0, 0)
    return date_from.strftime('%Y%m%d')

def user_history_view():
    date_id = gain_preN_day(d, date_diff)
    print('date_id',date_id)
    cmd = '''hive -e "set mapreduce.job.queuename=recom;
     select a.uid as uid,a.product_skn as product_skn,a.brand_id as brand_id,b.middle_sort_id as middle_sort_id,a.create_time as create_time,a.date_id as date_id 
from recom_yohobuy.user_history_view as a 
left outer join
(select product_skn,middle_sort_id from ods_erp_product.product where date_id=%s) as b
on a.product_skn=b.product_skn  where uid!=0  order by  uid ,create_time 
">/data/Algorithm_sun/rnn_brand_sort/model_data/train_data_%s.txt
    ''' % (date_id, date_id)
    os.system(cmd)
    return 0
def click_view_info():
    date_id = gain_preN_day(d, date_diff)
    print('date_id',date_id)
    cmd = '''hive -e "set mapreduce.job.queuename=recom;
     select a.user_log_acct as uid,a.product_skn as product_skn,a.brand_id as brand_id,b.middle_sort_id as middle_sort_id,a.time_stamp_click as
 create_time,a.date_id as date_id 
from  (
select * from recom_yohobuy.click_view_info where date_id=%s )   as a 
left outer join
(select product_skn,middle_sort_id from ods_erp_product.product where date_id=%s) as b
on a.product_skn=b.product_skn  where a.user_log_acct!=0  order by  uid ,create_time 
">/data/Algorithm_sun/rnn_brand_sort/model_data/click_brand_msort_data_%s.txt
    ''' % (date_id, date_id,date_id)
    os.system(cmd)
    return 0

if __name__ == '__main__':
    #user_history_view()
    click_view_info()

