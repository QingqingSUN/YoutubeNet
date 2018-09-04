import os
import os,time,sys,datetime
d = datetime.datetime.now()
date_diff = 2
def gain_preN_day(d, n):

    days_cnt = datetime.timedelta(days = n)


    # print(datetime.timedelta(days = n))
    day_from = d - days_cnt
    date_from = datetime.datetime(day_from.year, day_from.month, day_from.day, 0, 0, 0)
    return date_from.strftime('%Y%m%d')


def click_view_info():
    date_id = gain_preN_day(d, date_diff)
    print('date_id',date_id)
    cmd = '''hive -e "set mapreduce.job.queuename=recom;
     select user_log_acct,product_skn,time_stamp_click 
from recom_yohobuy.click_view_info where date_id=%s and user_log_acct is not null and user_log_acct !=0  
order by user_log_acct,time_stamp_click 
">/data/Algorithm_sun/YoutobeNet/youData/click_data_%s.txt
    ''' % (date_id,date_id)
    os.system(cmd)
    return 0

if __name__ == '__main__':

    click_view_info()

