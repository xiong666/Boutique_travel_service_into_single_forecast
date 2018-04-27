# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:52:43 2018

@author: xiong
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:35:48 2018

@author: admin
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 14:20:50 2017

@author: xiong
"""
########################################
#整体的用户数据进行训练
#######################################
import pandas as pd
import numpy as np
from datetime import date
import lightgbm as lgb
#import xgboost as xgb
from sklearn.cross_validation import train_test_split 
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn import metrics 
import pywt
import time

action_train = pd.read_csv('data/action_train.csv')
orderFuture_train = pd.read_csv('data/orderFuture_train.csv')
orderHistory_train = pd.read_csv('data/orderHistory_train.csv')
userComment_train = pd.read_csv('data/userComment_train.csv')
userProfile_train = pd.read_csv('data/userProfile_train.csv')
orderFuture_train.rename(columns={'orderType':'label'},inplace=True)

#==============================================================================
# feature6_train=pd.read_csv('data/windward_v1_train_time_trans_42.csv')
# feature6_test=pd.read_csv('data/windward_v1_test_time_trans_42.csv')
#==============================================================================
#==============================================================================
# feature_son_train=pd.read_csv('data/orderHistory_train_del_parent_son_order_unix_timestamp.csv')
# feature_son_test=pd.read_csv('data/orderHistory_test_del_parent_son_order_unix_timestamp.csv')
#==============================================================================
feature2_3_4_train=pd.read_csv('data/user_action_type_to_type_2_3_4_new_type_feature_train.csv')
feature2_3_4_test=pd.read_csv('data/user_action_type_to_type_2_3_4_new_type_feature_test.csv')
#==============================================================================
# feature_train_user_zuihouyicileibieshijiancha=pd.read_csv('data/feature_train_user_zuihouyicileibieshijiancha.csv')
# feature_test_user_zuihouyicileibieshijiancha=pd.read_csv('data/feature_test_user_zuihouyicileibieshijiancha.csv')
#==============================================================================

action_test = pd.read_csv('data/action_test.csv')
orderFuture_test = pd.read_csv('data/orderFuture_test.csv')
orderHistory_test = pd.read_csv('data/orderHistory_test.csv')
userComment_test = pd.read_csv('data/userComment_test.csv')
userProfile_test = pd.read_csv('data/userProfile_test.csv')


usert1_train = orderHistory_train[['userid','orderType']].groupby(['userid']).agg('mean').reset_index()
usert2_train = userComment_train[['userid','rating']].groupby(['userid']).agg('mean').reset_index()

action_train['actioncount'] = 1
usert3_train1 = action_train[['userid','actionType','actioncount']].groupby(['userid','actionType']).agg('count').reset_index()
usert3_train=usert3_train1.set_index(['userid','actionType']).unstack()
usert3_train['userid'] = usert3_train.index
usert3_train.columns = ['actioncount1','actioncount2','actioncount3','actioncount4','actioncount5','actioncount6','actioncount7','actioncount8','actioncount9','userid']

usert4_train = orderHistory_train[orderHistory_train.orderTime>orderHistory_train.orderTime.median()][['userid','orderType']].groupby(['userid']).agg('mean').reset_index()
usert4_train.rename(columns={'orderType':'orderType1'},inplace=True)

orderHistory = pd.concat([orderHistory_train,orderHistory_test])[['userid','orderType','country','city','continent']]
orderHistory['countrycount'] = 1
countrycount = orderHistory[['country','countrycount']].groupby(['country']).agg('sum').reset_index()
orderHistory.drop(["countrycount"], inplace=True, axis=1)
orderHistory = orderHistory.merge(countrycount, how='left', on=['country'])
orderHistory = orderHistory[orderHistory.countrycount>1]
usert5_train1 = orderHistory[['userid','orderType','country']].groupby(['userid','country']).agg('mean').reset_index()
usert5_train=usert5_train1.set_index(['userid','country']).unstack()
usert5_train.columns = ['country'+str(i+1) for i in range(usert5_train1.country.value_counts().shape[0])]
usert5_train['userid'] = usert5_train.index
##########################################################

#用户到旅游国家的精品率
usert51 = orderHistory[['userid','orderType','country']]
usert51['countrycount'] = 1
countrycount = usert51[['country','countrycount']].groupby(['country']).agg('sum').reset_index()
usert51.drop(["countrycount"], inplace=True, axis=1)
usert51 = usert51.merge(countrycount, how='left', on=['country'])
usert51 = usert51[usert51.countrycount>1]
usert51 = usert51[['userid','orderType','country']].groupby(['userid','country']).agg('mean').reset_index()
usert5=usert51.set_index(['userid','country']).unstack()
usert5.columns = ['country'+str(i+1) for i in range(usert51.country.value_counts().shape[0])]
usert5['userid'] = usert5.index

#用户到旅游城市的精品率
usert61 = orderHistory[['userid','orderType','city']]
usert61['citycount'] = 1
citycount = usert61[['city','citycount']].groupby(['city']).agg('sum').reset_index()
usert61.drop(["citycount"], inplace=True, axis=1)
usert61 = usert61.merge(citycount, how='left', on=['city'])
usert61 = usert61[usert61.citycount>30]
usert61 = usert61[['userid','orderType','city']].groupby(['userid','city']).agg('mean').reset_index()
usert6=usert61.set_index(['userid','city']).unstack()
usert6.columns = ['city'+str(i+1) for i in range(usert61.city.value_counts().shape[0])]
usert6['userid'] = usert6.index

#用户到旅游洲的精品率
usert71 = orderHistory[['userid','orderType','continent']]
usert71['continentcount'] = 1
continentcount = usert71[['continent','continentcount']].groupby(['continent']).agg('sum').reset_index()
usert71.drop(["continentcount"], inplace=True, axis=1)
usert71 = usert71.merge(continentcount, how='left', on=['continent'])
usert71 = usert71[usert71.continentcount>10]
usert71 = usert71[['userid','orderType','continent']].groupby(['userid','continent']).agg('mean').reset_index()
usert7=usert71.set_index(['userid','continent']).unstack()
usert7.columns = ['continent'+str(i+1) for i in range(usert71.continent.value_counts().shape[0])]
usert7['userid'] = usert7.index

########################################
#usert5_train = orderHistory_train[orderHistory_train.orderTime>orderHistory_train.orderTime.quantile(0.25)][['userid','orderType']].groupby(['userid']).agg('mean').reset_index()
#usert5_train.rename(columns={'orderType':'orderType2'},inplace=True)
#usert6_train = orderHistory_train[orderHistory_train.orderTime>orderHistory_train.orderTime.quantile(0.75)][['userid','orderType']].groupby(['userid']).agg('mean').reset_index()
#usert6_train.rename(columns={'orderType':'orderType3'},inplace=True)


trainset = userProfile_train.merge(usert1_train, how='left', on=['userid'])
trainset = trainset.merge(usert2_train, how='left', on=['userid'])
trainset = trainset.merge(usert3_train, how='left', on=['userid'])
trainset = trainset.merge(orderFuture_train, how='left', on=['userid'])
trainset = trainset.merge(usert4_train, how='left', on=['userid'])
trainset = trainset.merge(usert5_train, how='left', on=['userid'])
#trainset = trainset.merge(usert6_train, how='left', on=['userid'])


usert1_test = orderHistory_test[['userid','orderType']].groupby(['userid']).agg('mean').reset_index()
usert2_test = userComment_test[['userid','rating']].groupby(['userid']).agg('mean').reset_index() 
action_test['actioncount'] = 1
usert3_test1 = action_test[['userid','actionType','actioncount']].groupby(['userid','actionType']).agg('count').reset_index()
usert3_test=usert3_test1.set_index(['userid','actionType']).unstack()
usert3_test['userid'] = usert3_test.index
usert3_test.columns = ['actioncount1','actioncount2','actioncount3','actioncount4','actioncount5','actioncount6','actioncount7','actioncount8','actioncount9','userid']
usert4_test = orderHistory_test[orderHistory_test.orderTime>orderHistory_test.orderTime.median()][['userid','orderType']].groupby(['userid']).agg('mean').reset_index()
usert4_test.rename(columns={'orderType':'orderType1'},inplace=True)


testset = userProfile_test.merge(usert1_test, how='left', on=['userid'])
testset = testset.merge(usert2_test, how='left', on=['userid'])
testset = testset.merge(usert3_test, how='left', on=['userid'])
testset = testset.merge(usert4_test, how='left', on=['userid'])
testset = testset.merge(usert5_train, how='left', on=['userid'])

#==============================================================================
# ########################## make feature ################################
# 
# """
# 1.最近一次1的出现时间
# 2.最近一次2的出现时间
# 3.最近一次3的出现时间
# 4.最近一次4的出现时间
# 5.最近一次5的出现时间
# 6.最近一次6的出现时间
# 7.最近一次7的出现时间
# 8.最近一次8的出现时间
# 9.最近一次9的出现时间
# 
# 1.距离最近一次1的操作距离
# 2.距离最近一次2的操作距离
# 3.距离最近一次3的操作距离
# 4.距离最近一次4的操作距离
# 5.距离最近一次5的操作距离
# 6.距离最近一次6的操作距离
# 7.距离最近一次7的操作距离
# 8.距离最近一次8的操作距离
# 9.距离最近一次9的操作距离
# 
# """
# 
# 
# def actionType_sequence(action,trainset):
#     df = action.copy()
#     p = df[["userid", "actionType","actionTime"]].groupby("userid", as_index=False)
# 
#     length = len(p.size())
#     type_total = 9
#     min_distance = [[np.nan] * length for _ in range(type_total)]
#     min_time = [[np.nan] * length for _ in range(type_total)]
# 
#     for index,(name, group) in enumerate(p):
#         actionType = group["actionType"]
#         actionTime = group["actionTime"]
#         actionType = list(actionType)
#         actionTime = list(actionTime)
#         endTime = actionTime[-1]
# 
#         actionType = actionType[::-1]
#         actionTime = actionTime[::-1]
#         action_set = set(actionType)
#         for number in range(type_total):
#             if (number + 1) in action_set:
#                 loc = actionType.index(number + 1)
#                 min_distance[number][index] = loc
#                 min_time[number][index] = actionTime[loc]
#     result = p.first()
#     del result["actionType"]
#     del result["actionTime"]
#     for column in range(type_total):
#         result["actionType_recent_position_{}".format(column + 1)] = min_distance[column]
#     for column in range(type_total):
#         result["actionType_recent_time_{}".format(column + 1)] = min_time[column]
#     trainset=trainset.merge(result,how='left',on='userid')
#     return trainset
# trainset=actionType_sequence(action_train,trainset)
# testset=actionType_sequence(action_test,testset)
#==============================================================================
"""用户最后一单的年 月 日 小时 分特征
"""
def time_transform(x):
    time_local = time.localtime(x)
    date1=time.strftime("%Y%m%d%H%M%S", time_local)
    return date1

def orderaction_date(action,trainset):
    grouped = action[["userid", "actionTime"]].groupby("userid", as_index = False)
    result = grouped.max()
#     result["first_ordertime"] = result.actionTime.apply(lambda time: max_time - time)
#     result["end_ordertime"] = grouped.max().actionTime.apply(lambda time: max_time - time)
#     result["median_mediantime"] = grouped.median().actionTime.apply(lambda time:max_time - time)

    date2 = result.actionTime.apply(time_transform)
    result["year_last"] = date2.apply(lambda x:int(x[:4]))
    result["month_last"] = date2.apply(lambda x:int(x[4:6]))
    result["day_last"] = date2.apply(lambda x:int(x[6:8]))
    result["hour_last"] = date2.apply(lambda x:int(x[8:10]))
    result["minute_last"] = date2.apply(lambda x:int(x[10:12]))
    del result["actionTime"]
    trainset=trainset.merge(result,how='left',on='userid')
    return trainset
trainset=orderaction_date(action_train,trainset)
testset=orderaction_date(action_test,testset)


def end_day_action(actions,trainset,window = 1):
    df = actions.copy()
    df["yearmonthday"] = df["actionTime"].apply(time_transform).apply(lambda x: int(x[:8]))
    grouped = df[["userid", "yearmonthday", "actionType"]].groupby("userid", as_index=False)
    result = grouped.last()
    length = len(grouped)
    total = 5

    count_stack = [np.nan] * length
    max_stack = [np.nan] * length
    number_stack = [[np.nan] * length for _ in range(total)]
    rate_stack = [[np.nan] * length for _ in range(total)]
    for index, (name, group) in enumerate(grouped):
        yearmonthday = np.array(group["yearmonthday"])
        actionType = np.array(group["actionType"])
        actionType[actionType == 3] = 2
        actionType[actionType == 4] = 2
        actionType[actionType == 5] = 3
        actionType[actionType == 6] = 4
        actionType[actionType == 7] = 5
        actionType[actionType == 8] = 5
        actionType[actionType == 9] = 5

        end_day = np.max(yearmonthday)
        windowday = np.array([False] * len(yearmonthday))
        for w in range(window):
            windowday += yearmonthday == end_day
            end_day -= 1
        count_stack[index] = np.sum(windowday)
        if count_stack[index] == 0: continue
        max_stack[index] = np.max(actionType[windowday])
        for column in range(total):
            number_stack[column][index] = 1.0 * np.sum(actionType[windowday] == column + 1)  # / count_stack[index]
            rate_stack[column][index] = 1.0 * np.sum(actionType[windowday] == column + 1)/ count_stack[index]

    del result["actionType"]
    for column in range(total-1):
        result["enddaytype{}number".format(column + 1)] = number_stack[column]
        result["enddaytype{}rate".format(column + 1)] = rate_stack[column]

    result["enddaymaxtype"] = max_stack
    result["enddaycount"] = count_stack
    trainset=trainset.merge(result,how='left',on='userid')
    return trainset
trainset=end_day_action(action_train,trainset)
testset=end_day_action(action_test,testset)

def end_month_action(actions,trainset):
    df = actions.copy()
    df["yearmonth"] = df["actionTime"].apply(time_transform).apply(lambda x: int(x[:6]))
    grouped = df[["userid", "yearmonth", "actionType"]].groupby("userid", as_index=False)
    result = grouped.last()
    length = len(grouped)
    total = 9

    count_stack = [np.nan] * length
    mean_stack = [np.nan] * length
    rate_stack = [[np.nan] * length for _ in range(total)]
    for index, (name, group) in enumerate(grouped):
        yearmonth = np.array(group["yearmonth"])
        actionType = np.array(group["actionType"])
#         actionType[actionType == 3] = 2
#         actionType[actionType == 4] = 2
#         actionType[actionType == 5] = 3
#         actionType[actionType == 6] = 4
#         actionType[actionType == 7] = 5
#         actionType[actionType == 8] = 5
#         actionType[actionType == 9] = 5

        end_month = np.max(yearmonth)
        count_stack[index] = np.sum(yearmonth == end_month)
        if count_stack[index] == 0: continue
        mean_stack[index] = np.mean(actionType[yearmonth == end_month])
        for column in range(total-1):
            rate_stack[column][index] = 1.0 * np.sum(actionType[yearmonth == end_month] == column + 1) 
#             / count_stack[
#                 index]

    del result["actionType"]
#     for column in range(total):
#         result["endmonthtype{}rate".format(column + 1)] = rate_stack[column]

    result["endmonthmeantype"] = mean_stack
    result["endmonthcount"] = count_stack
    trainset=trainset.merge(result,how='left',on='userid')
    return trainset
trainset=end_month_action(action_train,trainset)
testset=end_month_action(action_test,testset)

############feature1 merge action的type9
def make_feature1(action_set,trainset):
    action1=action_set[action_set['actionType']==9].reset_index(drop=True)
    action1=action1.drop_duplicates(['userid']).reset_index(drop=True)
    action1.rename(columns={'actionType':'actionType9'},inplace=True)
    trainset=trainset.merge(action1[['userid','actionType9']],how='left',on='userid')
    return trainset
trainset=make_feature1(action_train,trainset)
testset=make_feature1(action_test,testset)

###########feature2 merge action平均时间间隔和平均type
def top(df,column='actionTime'):
    return df.sort_values(by=column)[:]
def chafen(df):
    return pd.DataFrame(np.diff(df,axis=0))
def daoshu1(df):
    return df[-1:].head(1)
def daoshu2(df):
    return df[-2:].head(1)
def daoshu3(df):
    return df[-3:].head(1)
def daoshu3_zong_jun(df):
    return df[-3:]
def daoshu4(df):
    return df[-4:].head(1)
def zhengshu1(df):
    return df.head(1)
def tianjia(df):
    return pd.Series(range(len(df)))
def make_feature2(action_set,trainset):
    action1 = action_set.sort_values(by=['userid','actionTime'], ascending=True)
    action2=action1.groupby('userid').apply(chafen)
    action2=action2.reset_index()
#==============================================================================
#     action3=action2.groupby('userid').agg('mean').reset_index()   #时间间隔的均值
#     action3.rename(columns={1:'mean_type',2:'mean_time_interval'},inplace=True)
#     action4=action2.groupby('userid').agg('std').reset_index()    #时间间隔的标准差
#     action4.rename(columns={2:'std_time_interval'},inplace=True)
#     action5=action2.groupby('userid').min().reset_index()         #时间间隔最小值
#     action5.rename(columns={2:'min_time_interval'},inplace=True)
#==============================================================================
    action6=action2.groupby('userid').tail(1)
    action6.rename(columns={1:'mowei_type',2:'mowei_time_interval'},inplace=True)   #时间间隔末尾值和最后一个type差值
    action7=(action2.groupby('userid').tail(2)).groupby('userid',as_index=False).nth(0)
    action7.rename(columns={1:'daoshu2_type',2:'daoshu2_time_interval'},inplace=True)   #时间间隔倒数第二个值和倒数第二个type差值
    action8=(action2.groupby('userid').tail(3)).groupby('userid',as_index=False).nth(0)
    action8.rename(columns={1:'daoshu3_type',2:'daoshu3_time_interval'},inplace=True)   #时间间隔倒数第三个值和倒数第三个type差值
    action9=(action2.groupby('userid').tail(4)).groupby('userid',as_index=False).nth(0)
    action9.rename(columns={2:'daoshu4_time_interval'},inplace=True)   #时间间隔倒数第四个值
    action10_1=action2.groupby('userid').tail(3)
    action10=action10_1.groupby('userid').agg('mean').reset_index()
    action10.rename(columns={2:'daoshu3_zong_jun_time_interval'},inplace=True)   #时间间隔倒数3个值的平均
    action11=action10_1.groupby('userid').agg('std').reset_index()
    action11.rename(columns={2:'daoshu3_zong_std_time_interval'},inplace=True)   #时间间隔倒数3个值的标准差
    action12=action2.groupby('userid').head(1)
    action12.rename(columns={1:'zhengshu1_type',2:'zhengshu1_time_interval'},inplace=True)   #时间间隔正数第一个值和正数第一个type差值
    
    action13=action1.groupby('userid').tail(1)
    action13.rename(columns={'actionType':'z_mowei_type','actionTime':'z_mowei_time'},inplace=True)
    action14=(action1.groupby('userid').tail(2)).groupby('userid',as_index=False).nth(0)
    action14.rename(columns={'actionType':'z_daoshu2_type','actionTime':'z_daoshu2_time'},inplace=True)
    action15=(action1.groupby('userid').tail(3)).groupby('userid',as_index=False).nth(0)
    action15.rename(columns={'actionType':'z_daoshu3_type','actionTime':'z_daoshu3_time'},inplace=True)
    action16=action1.groupby('userid').head(1)
    action16.rename(columns={'actionType':'z_zhengshu1_type','actionTime':'z_zhengshu1_time'},inplace=True)
    trainset=trainset.merge(action13[['userid','z_mowei_type','z_mowei_time']],how='left',on='userid')
    trainset=trainset.merge(action14[['userid','z_daoshu2_type','z_daoshu2_time']],how='left',on='userid')
    trainset=trainset.merge(action15[['userid','z_daoshu3_type','z_daoshu3_time']],how='left',on='userid')
    trainset=trainset.merge(action16[['userid','z_zhengshu1_type','z_zhengshu1_time']],how='left',on='userid')
    
    action18_1 = action1[['userid','actionType']].groupby(['userid']).agg('count').reset_index()
    action18_1.rename(columns={'actionType':'action_type_count'},inplace=True) 
    actionj =  action1[['userid','actionType','actionTime']].merge(action18_1, how='left', on=['userid'])
    action18_2 = actionj.groupby('userid').apply(tianjia).reset_index()
    action18_2.rename(columns={0:'actioncount'},inplace=True)  
    actionj =  pd.concat([actionj,action18_2[['actioncount']]],axis=1)
    

    actionk = actionj[actionj.actionType==9].groupby('userid').apply(daoshu1).reset_index(drop=True)
    actionk.rename(columns={'actioncount':'actionnow'},inplace=True) 
    actionj = actionj.merge(actionk[['userid','actionnow']], how='left', on=['userid'])
    action18_9 = actionj[actionj.actioncount>=actionj.actionnow]
    action18_9= action18_9[['userid','actionType','actionTime']].groupby('userid').apply(chafen).reset_index()
    #时间间隔的均值
    action18_9_9=action18_9.groupby('userid').agg('mean').reset_index()   
    action18_9_9.rename(columns={1:'9mean_type',2:'9mean_time_interval'},inplace=True)    #到最近的9的平均时间间隔
    action18_9_10=action18_9.groupby('userid').agg('min').reset_index()   
    action18_9_10.rename(columns={1:'9min_type',2:'9min_time_interval'},inplace=True)    #到最近的9的最小时间间隔
    action18_9_11=action18_9.groupby('userid').agg('max').reset_index()   
    action18_9_11.rename(columns={1:'9max_type',2:'9max_time_interval'},inplace=True)    #到最近的9的最大时间间隔
    action18_9_12=action18_9.groupby('userid').agg('std').reset_index()   
    action18_9_12.rename(columns={1:'9std_type',2:'9std_time_interval'},inplace=True)    #到最近的9的标准差时间间隔

    
    actionk5 = actionj[actionj.actionType==5].groupby('userid').apply(daoshu1).reset_index(drop=True)
    actionk5.rename(columns={'actioncount':'actionnow5'},inplace=True) 
    actionj = actionj.merge(actionk5[['userid','actionnow5']], how='left', on=['userid'])
    action18_5 = actionj[actionj.actioncount>=actionj.actionnow5]
    action18_5= action18_5[['userid','actionType','actionTime']].groupby('userid').apply(chafen).reset_index()
    #时间间隔的均值
    action18_5_5=action18_5.groupby('userid').agg('mean').reset_index()   
    action18_5_5.rename(columns={1:'5mean_type',2:'5mean_time_interval'},inplace=True)    #到最近的5的平均时间间隔
    action18_5_5_2=action18_5.groupby('userid').agg('min').reset_index()   
    action18_5_5_2.rename(columns={1:'5min_type',2:'5min_time_interval'},inplace=True)    #到最近的5的时间间隔最小值
    action18_5_5_3=action18_5.groupby('userid').agg('max').reset_index()   
    action18_5_5_3.rename(columns={1:'5max_type',2:'5max_time_interval'},inplace=True)    #到最近的5的时间间隔最大值
    action18_5_5_4=action18_5.groupby('userid').agg('std').reset_index()   
    action18_5_5_4.rename(columns={1:'5std_type',2:'5std_time_interval'},inplace=True)    #到最近的5的时间间隔标准差
    

    actionk3 = actionj[actionj.actionType==3].groupby('userid').apply(daoshu1).reset_index(drop=True)
    actionk3.rename(columns={'actioncount':'actionnow3'},inplace=True) 
    actionj = actionj.merge(actionk3[['userid','actionnow3']], how='left', on=['userid'])
    action18_3 = actionj[actionj.actioncount>=actionj.actionnow3]
    action18_3= action18_3[['userid','actionType','actionTime']].groupby('userid').apply(chafen).reset_index()
     #时间间隔的均值
    action18_3_3=action18_3.groupby('userid').agg('mean').reset_index()   
    action18_3_3.rename(columns={1:'3mean_type',2:'3mean_time_interval'},inplace=True)    #到最近的3的平均时间间隔
    action18_3_3_2=action18_3.groupby('userid').agg('min').reset_index()   
    action18_3_3_2.rename(columns={1:'3min_type',2:'3min_time_interval'},inplace=True)    #到最近的3的时间间隔最小值
    action18_3_3_3=action18_3.groupby('userid').agg('max').reset_index()   
    action18_3_3_3.rename(columns={1:'3max_type',2:'3max_time_interval'},inplace=True)    #到最近的3的时间间隔最大值
    action18_3_3_4=action18_3.groupby('userid').agg('std').reset_index()   
    action18_3_3_4.rename(columns={1:'3std_type',2:'3std_time_interval'},inplace=True)    #到最近的3的时间间隔标准差

    actionk6 = actionj[actionj.actionType==6].groupby('userid').apply(daoshu1).reset_index(drop=True)
    actionk6.rename(columns={'actioncount':'actionnow6'},inplace=True) 
    actionj = actionj.merge(actionk6[['userid','actionnow6']], how='left', on=['userid'])
    action18_6 = actionj[actionj.actioncount>=actionj.actionnow6]
    action18_6= action18_6[['userid','actionType','actionTime']].groupby('userid').apply(chafen).reset_index()
     #时间间隔的均值
    action18_6_6=action18_6.groupby('userid').agg('mean').reset_index()   
    action18_6_6.rename(columns={1:'6mean_type',2:'6mean_time_interval'},inplace=True)    #到最近的6的平均时间间隔
    action18_6_6_1=action18_6.groupby('userid').agg('std').reset_index()   
    action18_6_6_1.rename(columns={1:'6std_type',2:'6std_time_interval'},inplace=True)    #到最近的6的时间间隔标准差
    action18_6_6_2=action18_6.groupby('userid').agg('min').reset_index()   
    action18_6_6_2.rename(columns={1:'6min_type',2:'6min_time_interval'},inplace=True)    #到最近的6的时间间隔最小值
    action18_6_6_3=action18_6.groupby('userid').agg('max').reset_index()   
    action18_6_6_3.rename(columns={1:'6max_type',2:'6max_time_interval'},inplace=True)    #到最近的6的时间间隔最大值
    
    
    action19=action1.groupby('userid').apply(daoshu1).reset_index(drop=True)
    action19.rename(columns={'actionType':'action_type_daoshu1','actionTime':'action_time_daoshu1'},inplace=True)
    action19['action_time_daoshu1']=action19.action_time_daoshu1.max()-action19.action_time_daoshu1
    
#==============================================================================
#     trainset=trainset.merge(action3[['userid','mean_type','mean_time_interval']],how='left',on='userid')
#     trainset=trainset.merge(action4[['userid','std_time_interval']],how='left',on='userid')
#     trainset=trainset.merge(action5[['userid','min_time_interval']],how='left',on='userid')  
#==============================================================================
    trainset=trainset.merge(action6[['userid','mowei_type','mowei_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action7[['userid','daoshu2_type','daoshu2_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action8[['userid','daoshu3_type','daoshu3_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action9[['userid','daoshu4_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action10[['userid','daoshu3_zong_jun_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action11[['userid','daoshu3_zong_std_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action12[['userid','zhengshu1_type','zhengshu1_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action18_9_9[['userid','9mean_type','9mean_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action18_9_10[['userid','9min_type','9min_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action18_9_11[['userid','9max_type','9max_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action18_9_12[['userid','9std_type','9std_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action18_5_5[['userid','5mean_type','5mean_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action18_5_5_2[['userid','5min_type','5min_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action18_5_5_3[['userid','5max_type','5max_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action18_5_5_4[['userid','5std_type','5std_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action18_3_3[['userid','3mean_type','3mean_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action18_3_3_2[['userid','3min_type','3min_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action18_3_3_3[['userid','3max_type','3max_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action18_3_3_4[['userid','3std_type','3std_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action18_6_6[['userid','6mean_type','6mean_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action18_6_6_1[['userid','6std_type','6std_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action18_6_6_2[['userid','6min_type','6min_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action18_6_6_3[['userid','6max_type','6max_time_interval']],how='left',on='userid') 
    trainset=trainset.merge(action19[['userid','action_time_daoshu1']],how='left',on='userid') 
    return trainset
trainset=make_feature2(action_train,trainset)
testset=make_feature2(action_test,testset)


###########feature3 merge action中各个type的点击率
#==============================================================================
def make_feature3(action_set,trainset):
    action_set['actioncount'] = 1
    action1 = action_set[['userid','actionType','actioncount']].groupby(['userid','actionType']).agg('count').reset_index()
    action2=action_set[['userid','actioncount']].groupby(['userid']).agg('count').reset_index()
    action2.rename(columns={'actioncount':'sum_actioncount'},inplace=True)
    action1=action1.merge(action2, how='left', on=['userid'])
    action1['dianjilv']=action1['actioncount']/action1['sum_actioncount']
    action3=action1[action1['actionType']==1].reset_index()
    action3.rename(columns={'dianjilv':'dianjilv1'},inplace=True)
    action4=action1[action1['actionType']==2].reset_index()
    action4.rename(columns={'dianjilv':'dianjilv2'},inplace=True)
    action5=action1[action1['actionType']==3].reset_index()
    action5.rename(columns={'dianjilv':'dianjilv3'},inplace=True)
    action6=action1[action1['actionType']==4].reset_index()
    action6.rename(columns={'dianjilv':'dianjilv4'},inplace=True)
    action7=action1[action1['actionType']==5].reset_index()
    action7.rename(columns={'dianjilv':'dianjilv5'},inplace=True)
    action8=action1[action1['actionType']==6].reset_index()
    action8.rename(columns={'dianjilv':'dianjilv6'},inplace=True)
    action9=action1[action1['actionType']==7].reset_index()
    action9.rename(columns={'dianjilv':'dianjilv7'},inplace=True)
    action10=action1[action1['actionType']==8].reset_index()
    action10.rename(columns={'dianjilv':'dianjilv8'},inplace=True)
    action11=action1[action1['actionType']==9].reset_index()
    action11.rename(columns={'dianjilv':'dianjilv9'},inplace=True)
    
    trainset=trainset.merge(action3[['userid','dianjilv1']],how='left',on='userid')
    trainset=trainset.merge(action4[['userid','dianjilv2']],how='left',on='userid')
    trainset=trainset.merge(action5[['userid','dianjilv3']],how='left',on='userid')
    trainset=trainset.merge(action6[['userid','dianjilv4']],how='left',on='userid')
    trainset=trainset.merge(action7[['userid','dianjilv5']],how='left',on='userid')
    trainset=trainset.merge(action8[['userid','dianjilv6']],how='left',on='userid')
    trainset=trainset.merge(action9[['userid','dianjilv7']],how='left',on='userid')
    trainset=trainset.merge(action10[['userid','dianjilv8']],how='left',on='userid')
    trainset=trainset.merge(action11[['userid','dianjilv9']],how='left',on='userid')
    return trainset
trainset=make_feature3(action_train,trainset)
testset=make_feature3(action_test,testset)

#==============================================================================
# def last_record(df,cols):
#     df2 = df.sort_values(by=cols)[:]
#     return df2.tail(1)
# def make_feature4(action_set,trainset):
#     action = action_set[['userid','actionTime']].groupby('userid').apply(last_record,cols='actionTime').reset_index(drop=True)
#     action1 = action.rename(columns={'actionTime': 'LastActionTime'})  # 最后一次行动时间
#     trainset=trainset.merge(action1[['userid','LastActionTime']],how='left',on='userid')
#     return trainset
# trainset=make_feature4(action_train,trainset)
# testset=make_feature4(action_test,testset)
#==============================================================================

###########feature5 merge action中各个type的点击率
#==============================================================================
#==============================================================================
# def make_feature5(usert3_train,trainset):
#     usert3_train['2_4counthe_5lv']=(usert3_train['actioncount2']+usert3_train['actioncount3']+usert3_train['actioncount4'])/(usert3_train['actioncount5']+usert3_train['actioncount6']+usert3_train['actioncount7']+usert3_train['actioncount8']+usert3_train['actioncount9'])
#     trainset=trainset.merge(usert3_train[['userid','2_4counthe_5lv']], how='left', on=['userid'])
#     return trainset
# trainset=make_feature5(usert3_train,trainset)
# testset=make_feature5(usert3_test,testset)
# 
#==============================================================================
###########feature6 merge action中2-5之间的时间差值
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
# def make_feature6(action_set,trainset):
#     action56=action_set[(action_set.actionType>=2)|(action_set.actionType<=5)]
#     action56=action56.reset_index(drop=True)
#     action56_1=action56.groupby('userid').apply(top).reset_index(drop=True)
#     action56_2=action56_1.groupby('userid').apply(chafen)
#     action56_2=action56_2.reset_index()
#     action56_3=action56_2.groupby('userid').agg('mean').reset_index()   #5-6时间间隔的均值
#     action56_3.rename(columns={1:'56mean_type',2:'56mean_time_interval'},inplace=True)
#     action56_4=action56_2.groupby('userid').agg('std').reset_index()   #5-6时间间隔的标准差
#     action56_4.rename(columns={1:'56std_type',2:'56std_time_interval'},inplace=True)
#      
#  
#     trainset=trainset.merge(action56_3[['userid','56mean_time_interval']],how='left',on='userid')
#     trainset=trainset.merge(action56_4[['userid','56std_time_interval']],how='left',on='userid')
#     return trainset
# trainset=make_feature6(action_train,trainset)
# testset=make_feature6(action_test,testset)
#==============================================================================

#####feature7 在action_train的最后的10次type中，时间间隔的均值
def top10(df,n,column='actionTime'):
    return df.sort_values(by=column)[-n:]
def make_feature7(action_set,trainset):
    action1 = action_set.sort_values(by=['userid','actionTime'], ascending=True)
    df10=action1.groupby('userid').tail(10).reset_index(drop=True)
#==============================================================================
#     df['actioncount'] = 1
#     action1=df[['userid','actionType','actioncount']].groupby(['userid','actionType']).agg('count').reset_index()
#     action2=action1.groupby(['userid']).agg('max').reset_index()
#     trainset=trainset.merge(action2,how='left',on='userid')
#==============================================================================
    action2=df10.groupby('userid').apply(chafen)
    action2=action2.reset_index()
    action3=action2.groupby('userid').agg('mean').reset_index()   #倒数10个type的时间间隔的均值
    action3.rename(columns={1:'daoshu10_mean_type',2:'daoshu10_mean_time_interval'},inplace=True)
    action4=action2.groupby('userid').agg('std').reset_index()   #倒数10个type的时间间隔的标准差
    action4.rename(columns={1:'daoshu10_std_type',2:'daoshu10_std_time_interval'},inplace=True)
    action5=action2.groupby('userid').agg('min').reset_index()   #倒数10个type的时间间隔的最小值
    action5.rename(columns={1:'daoshu10_min_type',2:'daoshu10_min_time_interval'},inplace=True)
    action6=action2.groupby('userid').agg('max').reset_index()   #倒数10个type的时间间隔的最大值
    action6.rename(columns={1:'daoshu10_max_type',2:'daoshu10_max_time_interval'},inplace=True)
    action7=action2.groupby('userid').agg('median').reset_index()   #倒数10个type的时间间隔的中位数
    action7.rename(columns={1:'daoshu10_median_type',2:'daoshu10_median_time_interval'},inplace=True)
    
    df12=action1.groupby('userid').tail(12).reset_index(drop=True)
    action2_12=df12.groupby('userid').apply(chafen)
    action2_12=action2_12.reset_index()
    action3_12=action2_12.groupby('userid').agg('mean').reset_index()   #倒数12个type的时间间隔的均值
    action3_12.rename(columns={1:'daoshu12_mean_type',2:'daoshu12_mean_time_interval'},inplace=True)
    action4_12=action2_12.groupby('userid').agg('std').reset_index()   #倒数12个type的时间间隔的标准差
    action4_12.rename(columns={1:'daoshu12_std_type',2:'daoshu12_std_time_interval'},inplace=True)
    action5_12=action2_12.groupby('userid').agg('min').reset_index()   #倒数12个type的时间间隔的最小值
    action5_12.rename(columns={1:'daoshu12_min_type',2:'daoshu12_min_time_interval'},inplace=True)
    action6_12=action2_12.groupby('userid').agg('max').reset_index()   #倒数12个type的时间间隔的最大值
    action6_12.rename(columns={1:'daoshu12_max_type',2:'daoshu12_max_time_interval'},inplace=True)
    action7_12=action2_12.groupby('userid').agg('median').reset_index()   #倒数12个type的时间间隔的中位数
    action7_12.rename(columns={1:'daoshu12_median_type',2:'daoshu12_median_time_interval'},inplace=True)
    
    df11=action1.groupby('userid').tail(11).reset_index(drop=True)
    action2_11=df11.groupby('userid').apply(chafen)
    action2_11=action2_11.reset_index()
    action3_11=action2_11.groupby('userid').agg('mean').reset_index()   #倒数11个type的时间间隔的均值
    action3_11.rename(columns={1:'daoshu11_mean_type',2:'daoshu11_mean_time_interval'},inplace=True)
    action4_11=action2_11.groupby('userid').agg('std').reset_index()   #倒数11个type的时间间隔的标准差
    action4_11.rename(columns={1:'daoshu11_std_type',2:'daoshu11_std_time_interval'},inplace=True)
    action5_11=action2_11.groupby('userid').agg('min').reset_index()   #倒数11个type的时间间隔的最小值
    action5_11.rename(columns={1:'daoshu11_min_type',2:'daoshu11_min_time_interval'},inplace=True)
    action6_11=action2_11.groupby('userid').agg('max').reset_index()   #倒数11个type的时间间隔的最大值
    action6_11.rename(columns={1:'daoshu11_max_type',2:'daoshu11_max_time_interval'},inplace=True)
    action7_11=action2_11.groupby('userid').agg('median').reset_index()   #倒数11个type的时间间隔的中位数
    action7_11.rename(columns={1:'daoshu11_median_type',2:'daoshu11_median_time_interval'},inplace=True)
    
    df15=action1.groupby('userid').tail(15).reset_index(drop=True)
    action2_15=df15.groupby('userid').apply(chafen)
    action2_15=action2_15.reset_index()
    action3_15=action2_15.groupby('userid').agg('mean').reset_index()   #倒数15个type的时间间隔的均值
    action3_15.rename(columns={1:'daoshu15_mean_type',2:'daoshu15_mean_time_interval'},inplace=True)
    action4_15=action2_15.groupby('userid').agg('std').reset_index()   #倒数15个type的时间间隔的标准差
    action4_15.rename(columns={1:'daoshu15_std_type',2:'daoshu15_std_time_interval'},inplace=True)
    action5_15=action2_15.groupby('userid').agg('min').reset_index()   #倒数15个type的时间间隔的最小值
    action5_15.rename(columns={1:'daoshu15_min_type',2:'daoshu15_min_time_interval'},inplace=True)
    action6_15=action2_15.groupby('userid').agg('max').reset_index()   #倒数15个type的时间间隔的最大值
    action6_15.rename(columns={1:'daoshu15_max_type',2:'daoshu15_max_time_interval'},inplace=True)
    action7_15=action2_15.groupby('userid').agg('median').reset_index()   #倒数15个type的时间间隔的中位数
    action7_15.rename(columns={1:'daoshu15_median_type',2:'daoshu15_median_time_interval'},inplace=True)
    
    df14=action1.groupby('userid').tail(14).reset_index(drop=True)
    action2_14=df14.groupby('userid').apply(chafen)
    action2_14=action2_14.reset_index()
    action3_14=action2_14.groupby('userid').agg('mean').reset_index()   #倒数14个type的时间间隔的均值
    action3_14.rename(columns={1:'daoshu14_mean_type',2:'daoshu14_mean_time_interval'},inplace=True)
    action4_14=action2_14.groupby('userid').agg('std').reset_index()   #倒数14个type的时间间隔的标准差
    action4_14.rename(columns={1:'daoshu14_std_type',2:'daoshu14_std_time_interval'},inplace=True)
    action5_14=action2.groupby('userid').agg('min').reset_index()   #倒数14个type的时间间隔的最小值
    action5_14.rename(columns={1:'daoshu14_min_type',2:'daoshu14_min_time_interval'},inplace=True)
    action6_14=action2_14.groupby('userid').agg('max').reset_index()   #倒数14个type的时间间隔的最大值
    action6_14.rename(columns={1:'daoshu14_max_type',2:'daoshu14_max_time_interval'},inplace=True)
    action7_14=action2.groupby('userid').agg('median').reset_index()   #倒数14个type的时间间隔的中位数
    action7_14.rename(columns={1:'daoshu14_median_type',2:'daoshu14_median_time_interval'},inplace=True)
    
#==============================================================================
#     df20=action_set.groupby('userid').apply(top10,n=20).reset_index(drop=True)
#     action2_20=df20.groupby('userid').apply(chafen)
#     action2_20=action2_20.reset_index()
#     action3_20=action2_20.groupby('userid').agg('mean').reset_index()   #倒数20个type的时间间隔的均值
#     action3_20.rename(columns={1:'daoshu20_mean_type',2:'daoshu20_mean_time_interval'},inplace=True)
#==============================================================================
    
    
    trainset=trainset.merge(action3[['userid','daoshu10_mean_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action4[['userid','daoshu10_std_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action5[['userid','daoshu10_min_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action6[['userid','daoshu10_max_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action7[['userid','daoshu10_median_time_interval']],how='left',on='userid')
    
    trainset=trainset.merge(action3_12[['userid','daoshu12_mean_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action4_12[['userid','daoshu12_std_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action5_12[['userid','daoshu12_min_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action6_12[['userid','daoshu12_max_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action7_12[['userid','daoshu12_median_time_interval']],how='left',on='userid')
    
    trainset=trainset.merge(action3_11[['userid','daoshu11_mean_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action4_11[['userid','daoshu11_std_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action5_11[['userid','daoshu11_min_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action6_11[['userid','daoshu11_max_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action7_11[['userid','daoshu11_median_time_interval']],how='left',on='userid')
    
    trainset=trainset.merge(action3_15[['userid','daoshu15_mean_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action4_15[['userid','daoshu15_std_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action5_15[['userid','daoshu15_min_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action6_15[['userid','daoshu15_max_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action7_15[['userid','daoshu15_median_time_interval']],how='left',on='userid')
    
    trainset=trainset.merge(action3_14[['userid','daoshu14_mean_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action4_14[['userid','daoshu14_std_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action5_14[['userid','daoshu14_min_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action6_14[['userid','daoshu14_max_time_interval']],how='left',on='userid')
    trainset=trainset.merge(action7_14[['userid','daoshu14_median_time_interval']],how='left',on='userid')
    
   # trainset=trainset.merge(action3_20[['userid','daoshu20_mean_time_interval']],how='left',on='userid')
    return trainset
trainset=make_feature7(action_train,trainset)
testset=make_feature7(action_test,testset)
  
#####feature8为type5与6-9的差分值的type为正情况下的差分时间的均值和方差
#######6-8,6-9,7-9的差分值的type为正情况下的差分时间的均值和方差  
def make_feature8(action_set,trainset):
#==============================================================================
#     action1=action_set.groupby('userid').apply(top)
#     action1=action1.reset_index(drop=True)
#==============================================================================
    action1 = action_set.sort_values(by=['userid','actionTime'], ascending=True)
    
    action56=action1[(action1.actionType==5)|(action1.actionType==6)]
    action56=action56.reset_index(drop=True)
    action56_1=action56.sort_values(by=['userid','actionTime'], ascending=True)
    action56_2=action56_1.groupby('userid').apply(chafen)
    action56_2=action56_2.reset_index()
    action56_2.rename(columns={1:'actionType56',2:'56_time_interval'},inplace=True)
    action56_3=action56_2[action56_2.actionType56>0]
    action56_4=action56_3.groupby('userid').agg('mean').reset_index()
    action56_4.rename(columns={'56_time_interval':'56_time_interval_mean'},inplace=True)
    action56_5=action56_3.groupby('userid').agg('std').reset_index()
    action56_5.rename(columns={'56_time_interval':'56_time_interval_std'},inplace=True)
    action56_6=action56_3.groupby('userid').agg('min').reset_index()
    action56_6.rename(columns={'56_time_interval':'56_time_interval_min'},inplace=True)
    action56_7=action56_3.groupby('userid').agg('max').reset_index()
    action56_7.rename(columns={'56_time_interval':'56_time_interval_max'},inplace=True)
    
    action57=action1[(action1.actionType==5)|(action1.actionType==7)]
    action57=action57.reset_index(drop=True)
    action57_1=action57.sort_values(by=['userid','actionTime'], ascending=True)
    action57_2=action57_1.groupby('userid').apply(chafen)
    action57_2=action57_2.reset_index()
    action57_2.rename(columns={1:'actionType57',2:'57_time_interval'},inplace=True)
    action57_3=action57_2[action57_2.actionType57>0]
    action57_4=action57_3.groupby('userid').agg('mean').reset_index()
    action57_4.rename(columns={'57_time_interval':'57_time_interval_mean'},inplace=True)
    action57_5=action57_3.groupby('userid').agg('std').reset_index()
    action57_5.rename(columns={'57_time_interval':'57_time_interval_std'},inplace=True)
    action57_6=action57_3.groupby('userid').agg('min').reset_index()
    action57_6.rename(columns={'57_time_interval':'57_time_interval_min'},inplace=True)
    action57_7=action57_3.groupby('userid').agg('max').reset_index()
    action57_7.rename(columns={'57_time_interval':'57_time_interval_max'},inplace=True)
    
    action58=action1[(action1.actionType==5)|(action1.actionType==8)]
    action58=action58.reset_index(drop=True)
    action58_1=action58.sort_values(by=['userid','actionTime'], ascending=True)
    action58_2=action58_1.groupby('userid').apply(chafen)
    action58_2=action58_2.reset_index()
    action58_2.rename(columns={1:'actionType58',2:'58_time_interval'},inplace=True)
    action58_3=action58_2[action58_2.actionType58>0]
    action58_4=action58_3.groupby('userid').agg('mean').reset_index()
    action58_4.rename(columns={'58_time_interval':'58_time_interval_mean'},inplace=True)
    action58_5=action58_3.groupby('userid').agg('std').reset_index()
    action58_5.rename(columns={'58_time_interval':'58_time_interval_std'},inplace=True)
    action58_6=action58_3.groupby('userid').agg('min').reset_index()
    action58_6.rename(columns={'58_time_interval':'58_time_interval_min'},inplace=True)
    action58_7=action58_3.groupby('userid').agg('max').reset_index()
    action58_7.rename(columns={'58_time_interval':'58_time_interval_max'},inplace=True)
    
    action59=action1[(action1.actionType==5)|(action1.actionType==9)]
    action59=action59.reset_index(drop=True)
    action59_1=action59.sort_values(by=['userid','actionTime'], ascending=True)
    action59_2=action59_1.groupby('userid').apply(chafen)
    action59_2=action59_2.reset_index()
    action59_2.rename(columns={1:'actionType59',2:'59_time_interval'},inplace=True)
    action59_3=action59_2[action59_2.actionType59>0]
    action59_4=action59_3.groupby('userid').agg('mean').reset_index()
    action59_4.rename(columns={'59_time_interval':'59_time_interval_mean'},inplace=True)
    action59_5=action59_3.groupby('userid').agg('std').reset_index()
    action59_5.rename(columns={'59_time_interval':'59_time_interval_std'},inplace=True)
    action59_6=action59_3.groupby('userid').agg('min').reset_index()
    action59_6.rename(columns={'59_time_interval':'59_time_interval_min'},inplace=True)
    action59_7=action59_3.groupby('userid').agg('max').reset_index()
    action59_7.rename(columns={'59_time_interval':'59_time_interval_max'},inplace=True)
    
    action68=action1[(action1.actionType==6)|(action1.actionType==8)]
    action68=action68.reset_index(drop=True)
    action68_1=action68.sort_values(by=['userid','actionTime'], ascending=True)
    action68_2=action68_1.groupby('userid').apply(chafen)
    action68_2=action68_2.reset_index()
    action68_2.rename(columns={1:'actionType68',2:'68_time_interval'},inplace=True)
    action68_3=action68_2[action68_2.actionType68>0]
    action68_4=action68_3.groupby('userid').agg('mean').reset_index()
    action68_4.rename(columns={'68_time_interval':'68_time_interval_mean'},inplace=True)
    action68_5=action68_3.groupby('userid').agg('std').reset_index()
    action68_5.rename(columns={'68_time_interval':'68_time_interval_std'},inplace=True)
    action68_6=action68_3.groupby('userid').agg('min').reset_index()
    action68_6.rename(columns={'68_time_interval':'68_time_interval_min'},inplace=True)
    action68_7=action68_3.groupby('userid').agg('max').reset_index()
    action68_7.rename(columns={'68_time_interval':'68_time_interval_max'},inplace=True)
    
    action69=action1[(action1.actionType==6)|(action1.actionType==9)]
    action69=action69.reset_index(drop=True)
    action69_1=action69.sort_values(by=['userid','actionTime'], ascending=True)
    action69_2=action69_1.groupby('userid').apply(chafen)
    action69_2=action69_2.reset_index()
    action69_2.rename(columns={1:'actionType69',2:'69_time_interval'},inplace=True)
    action69_3=action69_2[action69_2.actionType69>0]
    action69_4=action69_3.groupby('userid').agg('mean').reset_index()
    action69_4.rename(columns={'69_time_interval':'69_time_interval_mean'},inplace=True)
    action69_5=action69_3.groupby('userid').agg('std').reset_index()
    action69_5.rename(columns={'69_time_interval':'69_time_interval_std'},inplace=True)
    action69_6=action69_3.groupby('userid').agg('min').reset_index()
    action69_6.rename(columns={'69_time_interval':'69_time_interval_min'},inplace=True)
    action69_7=action69_3.groupby('userid').agg('max').reset_index()
    action69_7.rename(columns={'69_time_interval':'69_time_interval_max'},inplace=True)
    
    action79=action1[(action1.actionType==7)|(action1.actionType==9)]
    action79=action79.reset_index(drop=True)
    action79_1=action79.sort_values(by=['userid','actionTime'], ascending=True)
    action79_2=action79_1.groupby('userid').apply(chafen)
    action79_2=action79_2.reset_index()
    action79_2.rename(columns={1:'actionType79',2:'79_time_interval'},inplace=True)
    action79_3=action79_2[action79_2.actionType79>0]
    action79_4=action79_3.groupby('userid').agg('mean').reset_index()
    action79_4.rename(columns={'79_time_interval':'79_time_interval_mean'},inplace=True)
    action79_5=action79_3.groupby('userid').agg('std').reset_index()
    action79_5.rename(columns={'79_time_interval':'79_time_interval_std'},inplace=True)
    action79_6=action79_3.groupby('userid').agg('min').reset_index()
    action79_6.rename(columns={'79_time_interval':'79_time_interval_min'},inplace=True)
    action79_7=action79_3.groupby('userid').agg('max').reset_index()
    action79_7.rename(columns={'79_time_interval':'79_time_interval_max'},inplace=True)
    
    trainset=trainset.merge(action56_4[['userid','56_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action56_5[['userid','56_time_interval_std']],how='left',on='userid')
    trainset=trainset.merge(action56_6[['userid','56_time_interval_min']],how='left',on='userid')
    trainset=trainset.merge(action56_7[['userid','56_time_interval_max']],how='left',on='userid')
    trainset=trainset.merge(action57_4[['userid','57_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action57_5[['userid','57_time_interval_std']],how='left',on='userid')
    trainset=trainset.merge(action57_6[['userid','57_time_interval_min']],how='left',on='userid')
    trainset=trainset.merge(action57_7[['userid','57_time_interval_max']],how='left',on='userid')
    trainset=trainset.merge(action58_4[['userid','58_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action58_5[['userid','58_time_interval_std']],how='left',on='userid')
    trainset=trainset.merge(action58_6[['userid','58_time_interval_min']],how='left',on='userid')
    trainset=trainset.merge(action58_7[['userid','58_time_interval_max']],how='left',on='userid')
    trainset=trainset.merge(action59_4[['userid','59_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action59_5[['userid','59_time_interval_std']],how='left',on='userid')
    trainset=trainset.merge(action59_6[['userid','59_time_interval_min']],how='left',on='userid')
    trainset=trainset.merge(action59_7[['userid','59_time_interval_max']],how='left',on='userid')
    
    trainset=trainset.merge(action68_4[['userid','68_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action68_5[['userid','68_time_interval_std']],how='left',on='userid')
    trainset=trainset.merge(action68_6[['userid','68_time_interval_min']],how='left',on='userid')
    trainset=trainset.merge(action68_7[['userid','68_time_interval_max']],how='left',on='userid')
    trainset=trainset.merge(action69_4[['userid','69_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action69_5[['userid','69_time_interval_std']],how='left',on='userid')
    trainset=trainset.merge(action69_6[['userid','69_time_interval_min']],how='left',on='userid')
    trainset=trainset.merge(action69_7[['userid','69_time_interval_max']],how='left',on='userid')
    trainset=trainset.merge(action79_4[['userid','79_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action79_5[['userid','79_time_interval_std']],how='left',on='userid')
    trainset=trainset.merge(action79_6[['userid','79_time_interval_min']],how='left',on='userid')
    trainset=trainset.merge(action79_7[['userid','79_time_interval_max']],how='left',on='userid')
    return trainset

trainset=make_feature8(action_train,trainset)
testset=make_feature8(action_test,testset)  


#####feature9为type1与5-9的差分值的type为正情况下的差分时间的均值和方差
##############type2-4与5的差分值的type为正情况下的差分时间的均值和方差
def make_feature9(action_set,trainset):
#==============================================================================
#     action1=action_set.groupby('userid').apply(top)
#     action1=action1.reset_index(drop=True)
#==============================================================================
    action1 = action_set.sort_values(by=['userid','actionTime'], ascending=True)
    
    action15=action1[(action1.actionType==1)|(action1.actionType==5)]
    action15=action15.reset_index(drop=True)
    action15_1=action15.sort_values(by=['userid','actionTime'], ascending=True)
    action15_2=action15_1.groupby('userid').apply(chafen)
    action15_2=action15_2.reset_index()
    action15_2.rename(columns={1:'actionType15',2:'15_time_interval'},inplace=True)
    action15_3=action15_2[action15_2.actionType15>0]
    action15_4=action15_3.groupby('userid').agg('mean').reset_index()
    action15_4.rename(columns={'15_time_interval':'15_time_interval_mean'},inplace=True)
    action15_5=action15_3.groupby('userid').agg('std').reset_index()
    action15_5.rename(columns={'15_time_interval':'15_time_interval_std'},inplace=True)
    action15_6=action15_3.groupby('userid').agg('min').reset_index()
    action15_6.rename(columns={'15_time_interval':'15_time_interval_min'},inplace=True)
    action15_7=action15_3.groupby('userid').agg('max').reset_index()
    action15_7.rename(columns={'15_time_interval':'15_time_interval_max'},inplace=True)
    
    action16=action1[(action1.actionType==1)|(action1.actionType==6)]
    action16=action16.reset_index(drop=True)
    action16_1=action16.sort_values(by=['userid','actionTime'], ascending=True)
    action16_2=action16_1.groupby('userid').apply(chafen)
    action16_2=action16_2.reset_index()
    action16_2.rename(columns={1:'actionType16',2:'16_time_interval'},inplace=True)
    action16_3=action16_2[action16_2.actionType16>0]
    action16_4=action16_3.groupby('userid').agg('mean').reset_index()
    action16_4.rename(columns={'16_time_interval':'16_time_interval_mean'},inplace=True)
    action16_5=action16_3.groupby('userid').agg('std').reset_index()
    action16_5.rename(columns={'16_time_interval':'16_time_interval_std'},inplace=True)
    action16_6=action16_3.groupby('userid').agg('min').reset_index()
    action16_6.rename(columns={'16_time_interval':'16_time_interval_min'},inplace=True)
    action16_7=action16_3.groupby('userid').agg('max').reset_index()
    action16_7.rename(columns={'16_time_interval':'16_time_interval_max'},inplace=True)
    
    action17=action1[(action1.actionType==1)|(action1.actionType==7)]
    action17=action17.reset_index(drop=True)
    action17_1=action17.sort_values(by=['userid','actionTime'], ascending=True)
    action17_2=action17_1.groupby('userid').apply(chafen)
    action17_2=action17_2.reset_index()
    action17_2.rename(columns={1:'actionType17',2:'17_time_interval'},inplace=True)
    action17_3=action17_2[action17_2.actionType17>0]
    action17_4=action17_3.groupby('userid').agg('mean').reset_index()
    action17_4.rename(columns={'17_time_interval':'17_time_interval_mean'},inplace=True)
    action17_5=action17_3.groupby('userid').agg('std').reset_index()
    action17_5.rename(columns={'17_time_interval':'17_time_interval_std'},inplace=True)
    action17_6=action17_3.groupby('userid').agg('min').reset_index()
    action17_6.rename(columns={'17_time_interval':'17_time_interval_min'},inplace=True)
    action17_7=action17_3.groupby('userid').agg('max').reset_index()
    action17_7.rename(columns={'17_time_interval':'17_time_interval_max'},inplace=True)
    
    action18=action1[(action1.actionType==1)|(action1.actionType==8)]
    action18=action18.reset_index(drop=True)
    action18_1=action18.sort_values(by=['userid','actionTime'], ascending=True)
    action18_2=action18_1.groupby('userid').apply(chafen)
    action18_2=action18_2.reset_index()
    action18_2.rename(columns={1:'actionType18',2:'18_time_interval'},inplace=True)
    action18_3=action18_2[action18_2.actionType18>0]
    action18_4=action18_3.groupby('userid').agg('mean').reset_index()
    action18_4.rename(columns={'18_time_interval':'18_time_interval_mean'},inplace=True)
    action18_5=action18_3.groupby('userid').agg('std').reset_index()
    action18_5.rename(columns={'18_time_interval':'18_time_interval_std'},inplace=True)
    action18_6=action18_3.groupby('userid').agg('min').reset_index()
    action18_6.rename(columns={'18_time_interval':'18_time_interval_min'},inplace=True)
    action18_7=action18_3.groupby('userid').agg('max').reset_index()
    action18_7.rename(columns={'18_time_interval':'18_time_interval_max'},inplace=True)
    
    action19=action1[(action1.actionType==1)|(action1.actionType==9)]
    action19=action19.reset_index(drop=True)
    action19_1=action19.sort_values(by=['userid','actionTime'], ascending=True)
    action19_2=action19_1.groupby('userid').apply(chafen)
    action19_2=action19_2.reset_index()
    action19_2.rename(columns={1:'actionType19',2:'19_time_interval'},inplace=True)
    action19_3=action19_2[action19_2.actionType19>0]
    action19_4=action19_3.groupby('userid').agg('mean').reset_index()
    action19_4.rename(columns={'19_time_interval':'19_time_interval_mean'},inplace=True)
    action19_5=action19_3.groupby('userid').agg('std').reset_index()
    action19_5.rename(columns={'19_time_interval':'19_time_interval_std'},inplace=True)
    action19_6=action19_3.groupby('userid').agg('min').reset_index()
    action19_6.rename(columns={'19_time_interval':'19_time_interval_min'},inplace=True)
    action19_7=action19_3.groupby('userid').agg('max').reset_index()
    action19_7.rename(columns={'19_time_interval':'19_time_interval_max'},inplace=True)
    
    action25=action1[(action1.actionType==2)|(action1.actionType==5)]
    action25=action25.reset_index(drop=True)
    action25_1=action25.sort_values(by=['userid','actionTime'], ascending=True)
    action25_2=action25_1.groupby('userid').apply(chafen)
    action25_2=action25_2.reset_index()
    action25_2.rename(columns={1:'actionType25',2:'25_time_interval'},inplace=True)
    action25_3=action25_2[action25_2.actionType25>0]
    action25_4=action25_3.groupby('userid').agg('mean').reset_index()
    action25_4.rename(columns={'25_time_interval':'25_time_interval_mean'},inplace=True)
    action25_5=action25_3.groupby('userid').agg('std').reset_index()
    action25_5.rename(columns={'25_time_interval':'25_time_interval_std'},inplace=True)
    action25_6=action25_3.groupby('userid').agg('min').reset_index()
    action25_6.rename(columns={'25_time_interval':'25_time_interval_min'},inplace=True)
    action25_7=action25_3.groupby('userid').agg('max').reset_index()
    action25_7.rename(columns={'25_time_interval':'25_time_interval_max'},inplace=True)
    
    action35=action1[(action1.actionType==3)|(action1.actionType==5)]
    action35=action35.reset_index(drop=True)
    action35_1=action35.sort_values(by=['userid','actionTime'], ascending=True)
    action35_2=action35_1.groupby('userid').apply(chafen)
    action35_2=action35_2.reset_index()
    action35_2.rename(columns={1:'actionType35',2:'35_time_interval'},inplace=True)
    action35_3=action35_2[action35_2.actionType35>0]
    action35_4=action35_3.groupby('userid').agg('mean').reset_index()
    action35_4.rename(columns={'35_time_interval':'35_time_interval_mean'},inplace=True)
    action35_5=action35_3.groupby('userid').agg('std').reset_index()
    action35_5.rename(columns={'35_time_interval':'35_time_interval_std'},inplace=True)
    action35_6=action35_3.groupby('userid').agg('min').reset_index()
    action35_6.rename(columns={'35_time_interval':'35_time_interval_min'},inplace=True)
    action35_7=action35_3.groupby('userid').agg('max').reset_index()
    action35_7.rename(columns={'35_time_interval':'35_time_interval_max'},inplace=True)
    
    action45=action1[(action1.actionType==4)|(action1.actionType==5)]
    action45=action45.reset_index(drop=True)
    action45_1=action45.sort_values(by=['userid','actionTime'], ascending=True)
    action45_2=action45_1.groupby('userid').apply(chafen)
    action45_2=action45_2.reset_index()
    action45_2.rename(columns={1:'actionType45',2:'45_time_interval'},inplace=True)
    action45_3=action45_2[action45_2.actionType45>0]
    action45_4=action45_3.groupby('userid').agg('mean').reset_index()
    action45_4.rename(columns={'45_time_interval':'45_time_interval_mean'},inplace=True)
    action45_5=action45_3.groupby('userid').agg('std').reset_index()
    action45_5.rename(columns={'45_time_interval':'45_time_interval_std'},inplace=True)
    action45_6=action45_3.groupby('userid').agg('min').reset_index()
    action45_6.rename(columns={'45_time_interval':'45_time_interval_min'},inplace=True)
    action45_7=action45_3.groupby('userid').agg('max').reset_index()
    action45_7.rename(columns={'45_time_interval':'45_time_interval_max'},inplace=True)
    
#==============================================================================
#     ######2，3，4之间的差分值
#     action234=action1[(action1.actionType==2)|(action1.actionType==3)|(action1.actionType==4)]
#     action234=action234.reset_index(drop=True)
#     action234_1=action234.sort_values(by=['userid','actionTime'], ascending=True)
#     action234_2=action234_1.groupby('userid').apply(chafen)
#     action234_2=action234_2.reset_index()
#     action234_2.rename(columns={1:'actionType234',2:'234_time_interval'},inplace=True)
#     action234_3=action234_2[action234_2.actionType234>0]
#     action234_4=action234_3.groupby('userid').agg('mean').reset_index()
#     action234_4.rename(columns={'234_time_interval':'234_time_interval_mean'},inplace=True)
#     action234_5=action234_3.groupby('userid').agg('std').reset_index()
#     action234_5.rename(columns={'234_time_interval':'234_time_interval_std'},inplace=True)
#     action234_6=action234_3.groupby('userid').agg('min').reset_index()
#     action234_6.rename(columns={'234_time_interval':'234_time_interval_min'},inplace=True)
#     action234_7=action234_3.groupby('userid').agg('max').reset_index()
#     action234_7.rename(columns={'234_time_interval':'234_time_interval_max'},inplace=True)
#==============================================================================
    
    trainset=trainset.merge(action15_4[['userid','15_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action15_5[['userid','15_time_interval_std']],how='left',on='userid')
#    trainset=trainset.merge(action15_6[['userid','15_time_interval_min']],how='left',on='userid')
#    trainset=trainset.merge(action15_7[['userid','15_time_interval_max']],how='left',on='userid')
    trainset=trainset.merge(action16_4[['userid','16_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action16_5[['userid','16_time_interval_std']],how='left',on='userid')
#    trainset=trainset.merge(action16_6[['userid','16_time_interval_min']],how='left',on='userid')
#    trainset=trainset.merge(action16_7[['userid','16_time_interval_max']],how='left',on='userid')
    trainset=trainset.merge(action17_4[['userid','17_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action17_5[['userid','17_time_interval_std']],how='left',on='userid')
#    trainset=trainset.merge(action17_6[['userid','17_time_interval_min']],how='left',on='userid')
#    trainset=trainset.merge(action17_7[['userid','17_time_interval_max']],how='left',on='userid')
    trainset=trainset.merge(action18_4[['userid','18_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action18_5[['userid','18_time_interval_std']],how='left',on='userid')
#    trainset=trainset.merge(action18_6[['userid','18_time_interval_min']],how='left',on='userid')
#    trainset=trainset.merge(action18_7[['userid','18_time_interval_max']],how='left',on='userid')
    trainset=trainset.merge(action19_4[['userid','19_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action19_5[['userid','19_time_interval_std']],how='left',on='userid')
#    trainset=trainset.merge(action19_6[['userid','19_time_interval_min']],how='left',on='userid')
#    trainset=trainset.merge(action19_7[['userid','19_time_interval_max']],how='left',on='userid')
    trainset=trainset.merge(action25_4[['userid','25_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action25_5[['userid','25_time_interval_std']],how='left',on='userid')
#    trainset=trainset.merge(action25_6[['userid','25_time_interval_min']],how='left',on='userid')
#    trainset=trainset.merge(action25_7[['userid','25_time_interval_max']],how='left',on='userid')
    trainset=trainset.merge(action35_4[['userid','35_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action35_5[['userid','35_time_interval_std']],how='left',on='userid')
#    trainset=trainset.merge(action35_6[['userid','35_time_interval_min']],how='left',on='userid')
#    trainset=trainset.merge(action35_7[['userid','35_time_interval_max']],how='left',on='userid')
    trainset=trainset.merge(action45_4[['userid','45_time_interval_mean']],how='left',on='userid')
    trainset=trainset.merge(action45_5[['userid','45_time_interval_std']],how='left',on='userid')
#    trainset=trainset.merge(action45_6[['userid','45_time_interval_min']],how='left',on='userid')
#    trainset=trainset.merge(action45_7[['userid','45_time_interval_max']],how='left',on='userid')
#==============================================================================
#     trainset=trainset.merge(action234_4[['userid','234_time_interval_mean']],how='left',on='userid')
#     trainset=trainset.merge(action234_5[['userid','234_time_interval_std']],how='left',on='userid')
#==============================================================================
    return trainset
trainset=make_feature9(action_train,trainset)
testset=make_feature9(action_test,testset)

def make_feature10(action_set,userProfile_train,trainset):
    last_action = action_set.sort_values(by='actionTime', ascending=False).drop_duplicates(['userid','actionType'])
    action_diff = userProfile_train[['userid']].copy()
    for i in range(9):
        temp = last_action[last_action['actionType']==(i+1)]
        action_diff = action_diff.merge(temp[['userid','actionTime']], 'left', 'userid')
        action_diff.rename(columns={'actionTime': 'last_{}'.format(i+1)}, inplace=True)
    feats = ['userid']
    for i in range(1,7):
        for j in range(i+1,8):
            feat = 'action_diff_{}_{}'.format(i,j)
            action_diff[feat] = action_diff['last_{}'.format(i)] - action_diff['last_{}'.format(j)]
            feats.append(feat)
    action_diff = action_diff[feats]
    trainset=trainset.merge(action_diff,how='left',on='userid')
    return trainset
trainset=make_feature10(action_train,userProfile_train,trainset)
testset=make_feature10(action_test,userProfile_test,testset)

#==============================================================================
# 归类之后的效果时间统计~
# 归类见代码中函数
# 
# 1.起始1的出现时间
# 2.起始2的出现时间
# 3.起始3的出现时间
# 4.起始4的出现时间
# 5.起始5的出现时间
# 6.起始6的出现时间
# 7.起始7的出现时间
# 8.起始8的出现时间
# 9.起始9的出现时间
# 
# 1.起始第一次1的操作距离
# 2.起始第一次2的操作距离
# 3.起始第一次3的操作距离
# 4.起始第一次4的操作距离
# 5.起始第一次5的操作距离
# 6.起始第一次6的操作距离
# 7.起始第一次7的操作距离
# 8.起始第一次8的操作距离
# 9.起始第一次9的操作距离
#==============================================================================
#==============================================================================
# def actionType_sequence_begin(action,trainset):
#     df = action.copy()
#     p = df[["userid", "actionType","actionTime"]].groupby("userid", as_index=False)
# 
#     length = len(p.size())
#     type_total = 6
#     min_distance = [[np.nan] * length for _ in range(type_total)]
#     min_time = [[np.nan] * length for _ in range(type_total)]
# 
#     for index,(name, group) in enumerate(p):
#         actionType = np.array(group["actionType"])
#         actionType[actionType==3] = 2
#         actionType[actionType==4] = 2
#         actionType[actionType==5] = 3
#         actionType[actionType==6] = 4
#         actionType[actionType==7] = 5
#         actionType[actionType==8] = 6
#         actionType[actionType==9] = 6
# 
#         actionTime = group["actionTime"]
#         actionType = list(actionType)
#         actionTime = list(actionTime)
# 
#         action_set = set(actionType)
#         for number in range(type_total):
#             if (number + 1) in action_set:
#                 loc = actionType.index(number + 1)
#                 min_distance[number][index] = loc
#                 min_time[number][index] = actionTime[loc]
#     result = p.first()
#     del result["actionType"]
#     del result["actionTime"]
#     for column in range(type_total):
#         result["actionType_begin_position_{}".format(column + 1)] = min_distance[column]
#     for column in range(type_total):
#         result["actionType_begin_time_{}".format(column + 1)] = min_time[column]
#     trainset=trainset.merge(result,how='left',on='userid')
#     return trainset
# trainset=actionType_sequence_begin(action_train,trainset)
# testset=actionType_sequence_begin(action_test,testset)
#==============================================================================
###############feature11 同一个type最后两次的时间差值
#线下：0.966468726549
#线上：0.9643

#==============================================================================
# def daoshu2_2(df):
#     return df[-2:]
# def make_feature11(action_set,trainset):
#     action1=action_train.groupby('userid').apply(top)
#     action1=action1.reset_index(drop=True)
#     
#     action5=action1[(action1.actionType==5)]
#     action5=action5.reset_index(drop=True)
#     action5_1=action5.groupby('userid').apply(daoshu2_2).reset_index(drop=True)
#     action5_2=action5_1.groupby('userid').apply(chafen)
#     action5_3=action5_2.reset_index()
#     action5_3.rename(columns={2:'5zuihouliang_time_interval'},inplace=True) 
#     
#     action6=action1[(action1.actionType==6)]
#     action6=action6.reset_index(drop=True)
#     action6_1=action6.groupby('userid').apply(daoshu2_2).reset_index(drop=True)
#     action6_2=action6_1.groupby('userid').apply(chafen)
#     action6_3=action6_2.reset_index()
#     action6_3.rename(columns={2:'6zuihouliang_time_interval'},inplace=True) 
#     
# #==============================================================================
# #     action7=action1[(action1.actionType==7)]
# #     action7=action7.reset_index(drop=True)
# #     action7_1=action7.groupby('userid').apply(daoshu2_2).reset_index(drop=True)
# #     action7_2=action7_1.groupby('userid').apply(chafen)
# #     action7_3=action7_2.reset_index()
# #     action7_3.rename(columns={2:'7zuihouliang_time_interval'},inplace=True) 
# #     
# #     action8=action1[(action1.actionType==8)]
# #     action8=action8.reset_index(drop=True)
# #     action8_1=action8.groupby('userid').apply(daoshu2_2).reset_index(drop=True)
# #     action8_2=action8_1.groupby('userid').apply(chafen)
# #     action8_3=action8_2.reset_index()
# #     action8_3.rename(columns={2:'8zuihouliang_time_interval'},inplace=True) 
# #     
# #==============================================================================
#     action9=action1[(action1.actionType==9)]
#     action9=action9.reset_index(drop=True)
#     action9_1=action9.groupby('userid').apply(daoshu2_2).reset_index(drop=True)
#     action9_2=action9_1.groupby('userid').apply(chafen)
#     action9_3=action9_2.reset_index()
#     action9_3.rename(columns={2:'9zuihouliang_time_interval'},inplace=True) 
#     
#     trainset=trainset.merge(action5_3[['userid','5zuihouliang_time_interval']],how='left',on='userid')
#     trainset=trainset.merge(action6_3[['userid','6zuihouliang_time_interval']],how='left',on='userid')
# #==============================================================================
# #     trainset=trainset.merge(action7_3[['userid','7zuihouliang_time_interval']],how='left',on='userid')
# #     trainset=trainset.merge(action8_3[['userid','8zuihouliang_time_interval']],how='left',on='userid')
# #==============================================================================
#     trainset=trainset.merge(action9_3[['userid','9zuihouliang_time_interval']],how='left',on='userid')
#     return trainset
# trainset=make_feature11(action_train,trainset)
# testset=make_feature11(action_test,testset)
#==============================================================================
#==============================================================================
#==============================================================================
    
###########feature5 orderHistory_train merge userProfile_train找年龄段，精品率等特征
#==============================================================================
#==============================================================================
# def make_feature5(orderHistory_train,userProfile_train,trainset):
#     userProfile_train['age']=userProfile_train['age'].replace(['80后','70后','60后','90后','00后'],[1980,1970,1960,1990,2000])
#     history=orderHistory_train.merge(userProfile_train, how='left', on=['userid'])
#==============================================================================

#==============================================================================
# ###############feature12 每个用户倒数一天的一些特征
# def make_feature12(action_set,trainset):
#     action1=action_train.groupby('userid').apply(top)
#     action1=action1.reset_index(drop=True)
#     action2=action1.groupby('userid').apply(daoshu1).reset_index(drop=True)
#     action2.rename(columns={'actionTime':'meige_user_zuihou_time'},inplace=True)
#     action1=action1.merge(action2[['userid','meige_user_zuihou_time']],how='left',on='userid')
#     action3=action1[action1.actionTime>(action1.meige_user_zuihou_time-86400)]
#     action3=action3.reset_index(drop=True)
#     action4=action3.groupby('userid').apply(chafen)
#     action4=action4.reset_index()
#     action4.rename(columns={1:'typecha',2:'time_interval'},inplace=True)
#     #action5=action4[action4.typecha>0]
#     action6=action4.groupby('userid').agg('mean').reset_index()
#     action6.rename(columns={'time_interval':'zuihou1_zheng_mean_interval_time'},inplace=True)
#     action7=action4.groupby('userid').agg('std').reset_index()
#     action7.rename(columns={'time_interval':'zuihou1_zheng_std_interval_time'},inplace=True)
#     action8=action4.groupby('userid').agg('min').reset_index()
#     action8.rename(columns={'time_interval':'zuihou1_zheng_min_interval_time'},inplace=True)
#     action9=action4.groupby('userid').agg('max').reset_index()
#     action9.rename(columns={'time_interval':'zuihou1_zheng_max_interval_time'},inplace=True)
#     trainset=trainset.merge(action6[['userid','zuihou1_zheng_mean_interval_time']],how='left',on='userid')
#     trainset=trainset.merge(action7[['userid','zuihou1_zheng_std_interval_time']],how='left',on='userid')
#     trainset=trainset.merge(action8[['userid','zuihou1_zheng_min_interval_time']],how='left',on='userid')
#     trainset=trainset.merge(action9[['userid','zuihou1_zheng_max_interval_time']],how='left',on='userid')
#     return trainset
# trainset=make_feature12(action_train,trainset)
# testset=make_feature12(action_test,testset)
#==============================================================================

#################feature13 应该是对type a,b a<b求a到下一个最近b的时间差的均值
def make_feature13(data,trainset):
    for i in range(1,9):
            for j in range(i+1,10):
                tmp=data[(data.actionType==i)|(data.actionType==j)]
                next_action=list(tmp.actionType)
                next_action.insert(0,0)
                next_action=next_action[0:-1]
                tmp['next_action']=next_action
                next_acttime=list(tmp.actionTime)
                next_acttime.insert(0,1505087866)
                next_acttime=next_acttime[0:-1]
                tmp['next_acttime']=next_acttime
                next_userid=list(tmp.userid)
                next_userid.insert(0,0)
                next_userid=next_userid[0:-1]
                tmp['next_userid']=next_userid
                tmp['res1']=tmp['actionType']-tmp['next_action']
                tmp['res2']=tmp['actionTime']-tmp['next_acttime']
                tmp['res3']=tmp['userid']-tmp['next_userid']
                tmp3=tmp[(tmp.res1==j-i)&(tmp.res3==0)]
                tmp3=tmp3.groupby('userid')['res2'].agg({'{}-{}-avg'.format(i,j):'mean','{}-{}-med'.format(i,j):'median','{}-{}-min'.format(i,j):'min'}).reset_index()
                trainset = trainset.merge(tmp3, 'left', 'userid')
    return trainset
trainset=make_feature13(action_train,trainset)
testset=make_feature13(action_test,testset)

#################feature14 每个user的action5,6,7,8的最后三个时间差(线下0.96707)
def make_feature14(action_train,trainset):
    action_train['timestamp'] = action_train['actionTime']
    action_train['rank2']=action_train.groupby(['userid','actionType'])['actionTime'].rank(ascending=False)
    for i in [5,6,7,8]:
        t = action_train[action_train['actionType'] == i]
        t = t[t['rank2']<5]
        t['diff'] = t.groupby('userid')['timestamp'].diff()
        t = t.groupby('userid')['diff'].agg({'{0}{0}-avg'.format(i):'mean',
                                             '{0}{0}-med'.format(i):'median',
                                             '{0}{0}-min'.format(i):'min',}).reset_index()
        trainset = trainset.merge(t,'left','userid')
    return trainset
trainset=make_feature14(action_train,trainset)
testset=make_feature14(action_test,testset)

#############feature15 观察用户之前是否有过精品记录############
#==============================================================================
def make_feature15(orderHistory_train,trainset):
    hh=orderHistory_train.groupby('userid')['orderType'].sum()
    hh=pd.DataFrame(hh)
    hh=hh.reset_index()
    hh['flag']=(hh.orderType>0)
    hh['flag']=hh['flag'].replace([False,True],[0,1])
    trainset=trainset.merge(hh[['userid','flag']],how='left',on='userid')
    return trainset
trainset=make_feature15(orderHistory_train,trainset)
testset=make_feature15(orderHistory_test,testset)
#==============================================================================
#==============================================================================
# #################feature15 每个user的action5,6,7,8的最开始的一个时间间隔
# def make_feature15(action_train,trainset):
#     action_train['timestamp'] = action_train['actionTime']
#     action_train['rank2']=action_train.groupby(['userid','actionType'])['actionTime'].rank(ascending=True)
#     for i in [5,6,7,8]:
#         t = action_train[action_train['actionType'] == i]
#         t = t[t['rank2']<5]
#         t['diff'] = t.groupby('userid')['timestamp'].diff()
#         t = t.groupby('userid')['diff'].agg({'{0}{0}-avg'.format(i):'mean',
#                                              '{0}{0}-med'.format(i):'median',
#                                              '{0}{0}-min'.format(i):'min',}).reset_index()
#         trainset = trainset.merge(t,'left','userid')
#     return trainset
# trainset=make_feature15(action_train,trainset)
# testset=make_feature15(action_test,testset)
#==============================================================================

###############feature16 leak就是order表里面没有，但是评论表里面有的 (线上0.96850,线下0.968035)
def gen_comment_feat(df, data1, data2):
    data1['have_comment'] = 1
    orders = data2.orderid.unique()
    trick = data1[~data1.orderid.isin(orders)]
    trick['trick'] = 1
    df = df.merge(data1[['userid','have_comment',]],'left','userid')
    df = df.merge(trick[['userid','trick']],'left','userid')
    return df
trainset = gen_comment_feat(trainset, userComment_train,orderHistory_train)
testset = gen_comment_feat(testset, userComment_test,orderHistory_test)

def gen_action_feat(df, data):
    data['timestamp'] = data['actionTime']
    data['actionTime'] = pd.to_datetime(data['actionTime'],unit='s')
    data = data.sort_values(by=['userid','timestamp'], ascending=True)
    data['time_diff'] = data.groupby('userid')['timestamp'].diff()
    data['rank'] = data.groupby('userid')['actionTime'].rank(ascending=False)
    end_date = data.groupby('userid')['timestamp'].max().reset_index().rename(columns={'timestamp':'end_date'})
    data = data.merge(end_date, 'left', 'userid')
    end_rank = data.groupby('userid')['rank'].min().reset_index().rename(columns={'rank':'end_rank'})
    data = data.merge(end_rank, 'left', 'userid')
    time_diff = data[data['rank']<30]
    time_diff_avg = time_diff.groupby('userid')['time_diff'].mean().reset_index().rename(columns={'time_diff':'time_diff_avg'})
    time_diff_var = time_diff.groupby('userid')['time_diff'].var().reset_index().rename(columns={'time_diff':'time_diff_var'})
    time_diff_min = time_diff.groupby('userid')['time_diff'].min().reset_index().rename(columns={'time_diff':'time_diff_min'})
    time_diff_max = time_diff.groupby('userid')['time_diff'].max().reset_index().rename(columns={'time_diff':'time_diff_max'})
    time_diff_median = time_diff.groupby('userid')['time_diff'].median().reset_index().rename(columns={'time_diff':'time_diff_median'})
    
    fuck = data[data['actionType']>5]
    fuck_avg = fuck.groupby('userid')['time_diff'].mean().reset_index().rename(columns={'time_diff':'fuck_avg'})
    fuck_var = fuck.groupby('userid')['time_diff'].var().reset_index().rename(columns={'time_diff':'fuck_var'})
    fuck_max = fuck.groupby('userid')['time_diff'].max().reset_index().rename(columns={'time_diff':'fuck_max'})
    fuck_median = fuck.groupby('userid')['time_diff'].median().reset_index().rename(columns={'time_diff':'fuck_median'})
    
    rank1 = data[data['rank']==1]
    rank1['1st_hour'] = rank1.actionTime.dt.hour
    rank1['1st_weekday'] = rank1.actionTime.dt.weekday
    rank1['1st_month'] = rank1.actionTime.dt.month
    rank1 = rank1.rename(columns={'actionType':'1st_action','time_diff':'1st_time_diff','timestamp':'last_timestamp'})
    rank1 = rank1[['userid','1st_action','1st_time_diff','1st_hour','1st_weekday','1st_month','last_timestamp',]]
    rank2 = data[data['rank']==2]
    rank2 = rank2[['userid','actionType','time_diff',]]
    rank2.columns = ['userid','2st_action','2st_time_diff',]
    rank3 = data[data['rank']==3]
    rank3 = rank3[['userid','actionType','time_diff',]]
    rank3.columns = ['userid','3st_action','3st_time_diff',]
    rank4 = data[data['rank']==4]
    rank4 = rank4[['userid','actionType','time_diff',]]
    rank4.columns = ['userid','4st_action','4st_time_diff',]
    rank5 = data[data['rank']==5]
    rank5 = rank5[['userid','actionType','time_diff']]
    rank5.columns = ['userid','5st_action','5st_time_diff']
    rank6 = data[data['rank']==6]
    rank6 = rank6[['userid','actionType',]]
    rank6.columns = ['userid','6st_action',]
    
    last_action = data.sort_values(by='timestamp', ascending=False).drop_duplicates(['userid','actionType'])
    last_action1 = last_action[last_action['actionType']==1]
    last_action1['last_action1'] = last_action1['end_date'] - last_action1['timestamp']
    last_action1 = last_action1[['userid','last_action1']]
    last_action7 = last_action[last_action['actionType']==7]
    last_action7['last_action7'] = last_action7['end_date'] - last_action7['timestamp']
    last_action7 = last_action7[['userid','last_action7']]
    
    first = data.drop_duplicates('userid')
    first = first[['userid','actionType','timestamp']]
    first.columns = ['userid','first_action','first_timestamp']
    
    action = data[['userid','actionType']]
    action_rate = pd.get_dummies(action,prefix='action',columns=['actionType'])
    action_rate['action_sum'] = 1
    action_rate = action_rate.groupby('userid', as_index=False).sum()
    action_rate.rename(columns={'action_1': 'action1_rate','action_2': 'action2_rate','action_3': 'action3_rate',
                                'action_4': 'action4_rate','action_5': 'action5_rate','action_6': 'action6_rate',
                                'action_7': 'action7_rate','action_8': 'action8_rate','action_9': 'action9_rate',}, inplace=True)
    for i in range(9):
        action_rate['action{}_rate'.format(i+1)] = action_rate['action{}_rate'.format(i+1)] / action_rate['action_sum']
    del action_rate['action_sum']

    data['date'] = data['actionTime'].dt.date
    active_days = data.groupby(['userid', 'date']).size().reset_index()
    active_days = active_days.groupby('userid').size().reset_index()
    active_days.rename(columns={0: 'active_days'}, inplace=True)
    
    _last_action = data.sort_values(by='rank', ascending=True).drop_duplicates(['userid','actionType'])
    _last_action7 = _last_action[_last_action['actionType']==7]
    _last_action7['_last_action7'] = _last_action7['rank'] - _last_action7['end_rank']
    _last_action7 = _last_action7[['userid','_last_action7']]
    
    df = df.merge(time_diff_avg, 'left', 'userid')
    df = df.merge(time_diff_var, 'left', 'userid')
    df = df.merge(time_diff_min, 'left', 'userid')
    df = df.merge(time_diff_max, 'left', 'userid')
    df = df.merge(time_diff_median, 'left', 'userid')
    
    df = df.merge(fuck_avg, 'left', 'userid')
    df = df.merge(fuck_var, 'left', 'userid')
    df = df.merge(fuck_max, 'left', 'userid')
    df = df.merge(fuck_median, 'left', 'userid')
    
    df = df.merge(rank1, 'left', 'userid')
    df = df.merge(rank2, 'left', 'userid')
    df = df.merge(rank3, 'left', 'userid')
    df = df.merge(rank4, 'left', 'userid')
    df = df.merge(rank5, 'left', 'userid')
    df = df.merge(rank6, 'left', 'userid')

    df = df.merge(last_action1, 'left', 'userid')
    df = df.merge(last_action7, 'left', 'userid')
    df = df.merge(first, 'left', 'userid')
        
    df = df.merge(action_rate, 'left', 'userid')
    
    df = df.merge(active_days, 'left', 'userid')
    
    df = df.merge(_last_action7, 'left', 'userid')
    return df
trainset = gen_action_feat(trainset, action_train)
testset = gen_action_feat(testset, action_test)

#==============================================================================
# ###########################################################################           
# def gen_order_feat(df, data):
#     data = data.sort_values(by=['userid','timestamp','orderType'],ascending=False).drop_duplicates(['userid','timestamp'])
#     
#     order = data[['userid','orderType','continent']]
#     order = pd.get_dummies(order,prefix='orderType',columns=['orderType'])
#     order = pd.get_dummies(order,prefix='continent',columns=['continent'])
#     order_sum = order.groupby('userid', as_index=False).sum()
#     order_sum.rename(columns={'orderType_0': 'orderType0_sum','orderType_1': 'orderType1_sum'}, inplace=True)
#     order_sum['order_sum'] = order_sum['orderType0_sum'] + order_sum['orderType1_sum']
#     del order_sum['orderType0_sum']
#     order_sum.rename(columns={'continent_0': 'continent0_sum','continent_1': 'continent1_sum',
#                               'continent_2': 'continent2_sum','continent_3': 'continent3_sum',
#                               'continent_4': 'continent4_sum','continent_5': 'continent5_sum'
#                              }, inplace=True)
# #==============================================================================
# #     del order_sum['continent2_sum']
# #     del order_sum['continent5_sum']
# #==============================================================================
#     
#     last_order = data.sort_values(by='timestamp', ascending=False).drop_duplicates('userid')
#     last_order['order_diff'] = (pd.to_datetime('2017-09-12') - last_order['orderTime'])
#     last_order['order_diff'] = last_order['order_diff'].dt.days
#     last_order.rename(columns={'timestamp': 'order_timestamp','continent': 'order_continent',
#                               'city': 'order_city','country': 'order_country',
#                               }, inplace=True)
#     last_order = last_order[['userid','order_diff','order_timestamp','order_city','order_country',
#                              'order_continent',]]
#     
# 
#     df = df.merge(order_sum,'left','userid')
#     df = df.merge(last_order,'left','userid')
#     return df
# 
# def gen_comment_feat(df, data1, data2):
#     data1['have_comment'] = 1
#     orders = data2.orderid.unique()
#     trick = data1[~data1.orderid.isin(orders)]
#     trick['trick'] = 1
#     df = df.merge(data1[['userid','have_comment',]],'left','userid')
#     df = df.merge(trick[['userid','trick']],'left','userid')
#     return df
# orderHistory_train['timestamp'] = orderHistory_train['orderTime']
# orderHistory_train['orderTime'] = pd.to_datetime(orderHistory_train['orderTime'],unit='s')
# trainset = gen_order_feat(trainset, orderHistory_train)
# trainset = gen_comment_feat(trainset, userComment_train,orderHistory_train)
# 
# orderHistory_test['timestamp'] = orderHistory_test['orderTime']
# orderHistory_test['orderTime'] = pd.to_datetime(orderHistory_test['orderTime'],unit='s')
# testset = gen_order_feat(testset, orderHistory_test)
# testset = gen_comment_feat(testset, userComment_test,orderHistory_test)
#==============================================================================

############匹配国家，城市，洲的精品率
trainset=trainset.merge(usert5,how='left',on='userid')
trainset=trainset.merge(usert6,how='left',on='userid')
trainset=trainset.merge(usert7,how='left',on='userid')
testset=testset.merge(usert5,how='left',on='userid')
testset=testset.merge(usert6,how='left',on='userid')
testset=testset.merge(usert7,how='left',on='userid')

###################merge第三方给的csv文档特征
#trainset=trainset.merge(feature6_train,how='left',on='userid')
#testset=testset.merge(feature6_test,how='left',on='userid')
#==============================================================================
# trainset=trainset.merge(feature_son_train,how='left',on='userid')
# testset=testset.merge(feature_son_test,how='left',on='userid')
#==============================================================================

#==============================================================================
# ######用户最后一次类别出现的时间差值(此特征代码python2.7环境下才能运行)
# def actionend_diff(action,trainset,pairs=None):
#     df = action.copy()
#     p = df[["userid", "actionType","actionTime"]].groupby("userid", as_index=False)
# 
#     length = len(p.size())
#     type_total = 6
#     min_distance = np.array([[np.nan] * length for _ in range(type_total)])
#     min_time = np.array([[np.nan] * length for _ in range(type_total)])
# 
#     for index,(name, group) in enumerate(p):
#         actionType = np.array(group["actionType"])
#         actionTime = group["actionTime"]
#         actionType[actionType==3] = 2
#         actionType[actionType==4] = 2
#         actionType[actionType==5] = 3
#         actionType[actionType==6] = 4
#         actionType[actionType==7] = 5
#         actionType[actionType==8] = 6
#         actionType[actionType==9] = 6
#         actionType = list(actionType)
#         actionTime = list(actionTime)
#         endTime = actionTime[-1]
# 
#         actionType = actionType[::-1]
#         actionTime = actionTime[::-1]
#         action_set = set(actionType)
#         for number in range(type_total):
#             if (number + 1) in action_set:
#                 loc = actionType.index(number + 1)
#                 min_distance[number][index] = loc
#                 min_time[number][index] = actionTime[loc]
#     result = p.first()
#     del result["actionType"]
#     del result["actionTime"]
#     
#     for i in range(type_total-1):
#         for j in range(type_total-1):
#             result["typeend{}_{}diff".format(i+1,j+1)] =  min_time[i]-min_time[j]
# #     for i in range(type_total):
# #         for j in range(type_total):
#             result["typeend{}_{}dist".format(i+1,j+1)] = min_distance[i] - min_distance[j]
#     trainset=trainset.merge(result,how='left',on='userid')
#     return trainset
# trainset=actionend_diff(action_train,trainset,pairs=None)
# testset=actionend_diff(action_test,testset,pairs=None)
# ##################################################################
#==============================================================================


#################类别转移概率矩阵
def type_type(action,trainset):
    grouped = action[["userid", "actionType"]].groupby("userid", as_index=False)
    length = len(grouped.size())

    total = 5
    continue_type = [[[np.nan] * length for _ in range(total)] for _ in range(total)]
    for index,(name, group) in enumerate(grouped):
        actionType = np.array(group["actionType"])
        actionType[actionType==2] = 2
        actionType[actionType==3] = 2
        actionType[actionType==4] = 2
        actionType[actionType==5] = 3
        actionType[actionType==6] = 4
        actionType[actionType==7] = 5
        actionType[actionType==8] = 5
        actionType[actionType==9] = 5
        t = actionType[:-1]
        type_to = actionType[1:]
        if len(type_to) == 0:
            continue
        for i in range(total):
            for j in range(total):
                continue_type[i][j][index] = 1.0*np.sum((type_to==j+1)&(t==i+1))/len(type_to)

    result = grouped.first()
    del result["actionType"]
    for i in range(total-1):
        for j in range(total-1):
            result["type_{}_to_type_{}".format(i+1,j+1)] = continue_type[i][j]
    trainset=trainset.merge(result,how='left',on='userid')
    return trainset
trainset=type_type(action_train,trainset)
testset=type_type(action_test,testset)

#==============================================================================
# #############################特征8:把2,3,4和在一起变成10，计算其与其他类别的特征(此代码只能在python2.7环境下运行)####################################
# def init_dict():
#     type_dict={}
#     for i in [1,5,6,7,8,9,10]:
#         if str(i) not in type_dict:
#             type_dict[str(i)]={}
#             for j in [1,5,6,7,8,9,10]:
#                 if str(j) not in type_dict[str(i)]:
#                     #type_dict[str(i)][str(j)]={"max":"","min":"","var":[0],"mean":[0]}
# 
#                     type_dict[str(i)][str(j)] = {"max": "", "min": "",}
# 
#     return type_dict
# 
# def init_by_type_dict(_dict):
#     user_dict = {}
#     user_dict["userid"] = []
#     for i, value_i in _dict.items():
#         for j, value_j in value_i.items():
#             user_dict["start_" + str(j) + "_to_" + str(i) + "_max"] = []
#             user_dict["start_" + str(j) + "_to_" + str(i) + "_min"] = []
#             #user_dict["to_" + str(i) + "_start_" + str(j) + "_min"] = []
# 
# 
# 
#             # user_dict["to_" + str(i) + "_start_" + str(j) + "_var"] = []
#             #user_dict["to_" + str(i) + "_start_" + str(j) + "_mean"] = []
# 
#     return user_dict
# 
# 
# def push_data(data,value,userid):
#     data["userid"].append(userid)
#     for i in [1,5,6,7,8,9,10]:
#         for j in [1,5,6,7,8,9,10]:
#             data["start_" + str(j) + "_to_" + str(i) + "_max"].append(value[str(i)][str(j)]["max"])
#             data["start_" + str(j) + "_to_" + str(i) + "_min"].append(value[str(i)][str(j)]["min"])
# 
#             # data["to_" + str(i) + "_start_" + str(j) + "_var"].append(np.array(value[str(i)][str(j)]["var"]).var())
#             # data["to_" + str(i) + "_start_" + str(j) + "_mean"].append(np.array(value[str(i)][str(j)]["mean"]).mean())
# 
# def get_2_3_4_new_type_feature(train,trainset):
#     train["actionType"]=train["actionType"].astype("str")
#     train["actionType"][train["actionType"]=="2"]="10"
#     train["actionType"][train["actionType"]=="3"]="10"
#     train["actionType"][train["actionType"]=="4"]="10"
#     #print(train[train["actionType"]=="10"].head(10))
#     action_type_list = train.groupby(["userid"])["actionType"].agg(
#         {"action_type_list": lambda x: ",".join(x)}).reset_index()
#     train.actionTime = train.actionTime.astype("str")
#     action_time_list = train.groupby(["userid"])["actionTime"].agg(
#         {"action_time_list": lambda x: ",".join(x)}).reset_index()
#     action_list = pd.merge(action_type_list, action_time_list, on="userid", how="left")
# 
#     count = 0
#     user_dict = {}
#     for item in action_list.values:
#         if count % 1000 == 0:
#             print(count)
#         count += 1
#         type_dict = init_dict()
#         types = item[1].split(",")
#         times = item[2].split(",")
#         type_dict=init_dict()
# 
#         if len(types)>1:
#             for i in range(1,len(times)):
#                 for j in range(0,i):
#                     to_type=types[i]
#                     start_type=types[j]
#                     type_time_dis=int(times[i])-int(times[j])
# 
#                     if type_dict[to_type][start_type]["max"] == "":
#                         type_dict[to_type][start_type]["max"] = type_time_dis
#                     elif type_dict[to_type][start_type]["max"]<type_time_dis:
#                         type_dict[to_type][start_type]["max"]=type_time_dis
#                     if type_dict[to_type][start_type]["min"]=="":
#                         type_dict[to_type][start_type]["min"] = type_time_dis
#                     elif type_dict[to_type][start_type]["min"]>type_time_dis:
#                         type_dict[to_type][start_type]["min"]=type_time_dis
# 
#                     # type_dict[to_type][start_type]["var"].append(type_time_dis)
#                     # type_dict[to_type][start_type]["mean"].append(type_time_dis)
# 
# 
#         if str(item[0]) not in user_dict:
#             user_dict[str(item[0])] = type_dict
# 
# 
# 
# 
#     data=init_by_type_dict(init_dict())
#     for user,value in user_dict.items():
#         push_data(data,value,user)
# 
#     result=pd.DataFrame(data)
#     trainset=trainset.merge(result,how='left',on='userid')
#     return trainset
# trainset=get_2_3_4_new_type_feature(action_train,trainset)
# testset=get_2_3_4_new_type_feature(action_test,testset)
# ###################################
#==============================================================================
#==============================================================================
#==============================================================================

#####用户最后一次类别出现的时间差值是负特征
# # trainset=trainset.merge(feature_train_user_zuihouyicileibieshijiancha,how='left',on='userid')
# # testset=testset.merge(feature_test_user_zuihouyicileibieshijiancha,how='left',on='userid')
#==============================================================================
#==============================================================================
trainset=trainset.merge(feature2_3_4_train,how='left',on='userid')
testset=testset.merge(feature2_3_4_test,how='left',on='userid')

##########feature18 小波变换  线下：0.96917  线上：0.97000
def wt(row):
    a = int(row['1st_action'])
    b = int(row['2st_action'])
    c = int(row['3st_action'])
    d = int(row['4st_action'])
    cA, cD = pywt.dwt([a, b, c, d], 'db2')
    return pd.Series([cA[0],cA[1],cD[0],cD[1]])
    
aa = trainset.fillna(0)
trainset[['ca0','ca1','cd0','cd1']] = aa.apply(wt, axis=1)

aa = testset.fillna(0)
testset[['ca0','ca1','cd0','cd1']] = aa.apply(wt, axis=1)

#==============================================================================

trainset.userid = trainset.userid.astype('category')
trainset.gender = trainset.gender.astype('category')
trainset.province = trainset.province.astype('category')
#==============================================================================
# trainset.province_x = trainset.province_x.astype('category')
#==============================================================================
# trainset.city = trainset.city.astype('category')
# trainset.country = trainset.country.astype('category')
# trainset.continent = trainset.continent.astype('category')
#==============================================================================
#==============================================================================
trainset.age = trainset.age.astype('category')
testset.userid = testset.userid.astype('category')
testset.gender = testset.gender.astype('category')
testset.province = testset.province.astype('category')
#==============================================================================
# testset.province_x = testset.province_x.astype('category')
#==============================================================================
# testset.city = testset.city.astype('category')
# testset.country = testset.country.astype('category')
# testset.continent = testset.continent.astype('category')
#==============================================================================
#==============================================================================
testset.age = testset.age.astype('category')


train_features=[x for x in trainset.columns if x not in ['label','userid']]  
p_test = np.zeros(shape=[len(testset)])
test_auc=0
#X_train,test_X, y_train, test_y = train_test_split(trainset[train_features],trainset['label'],test_size = 0.2,random_state = 0) 
#==============================================================================
# kf = KFold(n_splits=20) #线下：0.96917  线上：0.97000
# for train_indices ,validate_indices in kf.split(trainset) : 
#     train_data = lgb.Dataset(trainset[train_features].loc[train_indices,:],label=trainset.loc[train_indices,'label'])
#     val_data = lgb.Dataset(trainset[train_features].loc[validate_indices,:],label=trainset.loc[validate_indices,'label'])
#==============================================================================
    
kf = StratifiedKFold(n_splits=20)  #线下：0.96906  线上:0.97009
                                   #20180131 16:43 乱加特征后 线下：0.96920 线上:0.97020
for train_indices ,validate_indices in kf.split(trainset[train_features].values,trainset['label'].values) : 
    train_data = lgb.Dataset(trainset[train_features].loc[train_indices,:],label=trainset.loc[train_indices,'label'])
    val_data = lgb.Dataset(trainset[train_features].loc[validate_indices,:],label=trainset.loc[validate_indices,'label'])

#xgb_params = {
#    'eta': 0.037,
#    'max_depth': 5,
#    'subsample': 0.80,
#    'objective': 'reg:linear',
#    'eval_metric': 'mae',
#    'lambda': 0.8,   
#    'alpha': 0.4, 
#    'silent': 1
#}
#
#dtrain = xgb.DMatrix(X_train, y_train)
#dvalid = xgb.DMatrix(test_X, test_y)
#dtest = xgb.DMatrix(testA_industry)
#
#num_boost_rounds = 250
#
#model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
#
#xgb_pred1 = model.predict(dtest)


#lgb_train = lgb.Dataset(trainset[train_features], label=trainset['label'])
#lgb_train = lgb.Dataset(X_train, label=y_train)
#lgb_test = lgb.Dataset(test_X, label=test_y)

    params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'metric_freq':1,
    'is_training_metric':'false',
    'max_bin':255,
    'num_leaves': 64,
    'learning_rate': 0.03,
    'tree_learner':'serial',
    'feature_fraction': 0.85,
    'feature_fraction_seed':2,
    'bagging_fraction': 0.85,
#==============================================================================
#     'bagging_freq': 5,
#     'min_date_in_leaf':100,
#     'min_sum_hessian_in_leaf':100,
#     'max_depth':6,
#==============================================================================
    'early_stopping_round':100,
    'verbose':0,
    'is_unbalance':'true'
    }  

    clf = lgb.train(params, train_data,1200,valid_sets={train_data,val_data})
    p_val=clf.predict(trainset[train_features].loc[validate_indices,:])
    test_auc+=metrics.roc_auc_score(trainset.loc[validate_indices,'label'],p_val)
    p_test += clf.predict(testset[train_features])
print(test_auc/20)
p_test=p_test/20
submission = pd.DataFrame(testset['userid'])
submission['orderType'] = p_test
submission.to_csv('huanbaoche_result_cv20180129.csv',index=None)
