# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 13:38:38 2018

@author: xiong
"""

import pandas as pd
import numpy as np
from datetime import date
import lightgbm as lgb
import pywt
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn import metrics
import pickle
trainset = pickle.load(open('data/xiong_trainset.pkl', 'rb'))
testset = pickle.load(open('data/xiong_testset.pkl', 'rb'))

trainset.gender=trainset.gender.apply(lambda x:1 if x=='男' else x)
trainset.gender=trainset.gender.apply(lambda x:0 if x=='女' else x)
trainset.age=trainset.age.apply(lambda x:0 if x=='00后' else x)
trainset.age=trainset.age.apply(lambda x:1 if x=='90后' else x)
trainset.age=trainset.age.apply(lambda x:2 if x=='80后' else x)
trainset.age=trainset.age.apply(lambda x:3 if x=='70后' else x)
trainset.age=trainset.age.apply(lambda x:4 if x=='60后' else x)
#
testset.gender=testset.gender.apply(lambda x:1 if x=='男' else x)
testset.gender=testset.gender.apply(lambda x:0 if x=='女' else x)
testset.age=testset.age.apply(lambda x:0 if x=='00后' else x)
testset.age=testset.age.apply(lambda x:1 if x=='90后' else x)
testset.age=testset.age.apply(lambda x:2 if x=='80后' else x)
testset.age=testset.age.apply(lambda x:3 if x=='70后' else x)
testset.age=testset.age.apply(lambda x:4 if x=='60后' else x)

trainset['gender'] = trainset['gender'].astype('float64')
trainset['age'] = trainset['age'].astype('float64')
testset['gender'] = testset['gender'].astype('float64')
testset['age'] = testset['age'].astype('float64')
train_features=[x for x in trainset.columns if x not in ['label','userid','province']]
xiong_train_X=trainset[train_features].values
xiong_train_Y=trainset['label'].values
xiong_test_X=testset[train_features].values
output1 = open('data/xiong_train_X.pkl', 'wb')
pickle.dump(xiong_train_X, output1)
output1.close()
output2 = open('data/xiong_train_Y.pkl', 'wb')
pickle.dump(xiong_train_Y, output2)
output2.close()
output3 = open('data/xiong_test_X.pkl', 'wb')
pickle.dump(xiong_test_X, output3)
output3.close()

