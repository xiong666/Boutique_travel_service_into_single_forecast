# -*- coding: utf-8 -*-

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import auc,roc_curve,accuracy_score,roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.preprocessing import MinMaxScaler,StandardScaler,scale

trainset = pickle.load(open('data/xiong_train_X.pkl','rb'))
train_Y = pickle.load(open('data/xiong_train_Y.pkl','rb'))
testset = pickle.load(open('data/xiong_test_X.pkl','rb'))
#类别特征转类型
#trainset.gender = trainset.gender.astype('category')
#trainset.action_type_daoshu1 = trainset.action_type_daoshu1.astype('category')
#trainset.action_type_daoshu2 = trainset.action_type_daoshu2.astype('category')
#trainset.action_type_daoshu3 = trainset.action_type_daoshu3.astype('category')
#trainset.action_type_zhengshu1 = trainset.action_type_zhengshu1.astype('category')
#trainset.action_type_daoshu4 = trainset.action_type_daoshu4.astype('category')
#
#
#testset.gender = testset.gender.astype('category')
#testset.action_type_daoshu1 = testset.action_type_daoshu1.astype('category')
#testset.action_type_daoshu2 = testset.action_type_daoshu2.astype('category')
#testset.action_type_daoshu3 = testset.action_type_daoshu3.astype('category')
#testset.action_type_zhengshu1 = testset.action_type_zhengshu1.astype('category')
#testset.action_type_daoshu4 = testset.action_type_daoshu4.astype('category')

train_X=trainset.values
train_Y=train_Y.values
test_X=testset.values
# 基模型参数，调适合自己特征的

lgb_params = {
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
    'is_unbanlance':'true',
    'num_iterations':1200
    }  

xgb_params = {
     'learning_rate': 0.03,
     'max_depth': 7,
     'subsample': 0.85,
     'objective': 'binary:logistic',
     #'eval_metric': 'roc_auc',
     #'lambda': 0.8,
     #min_child_weight':3, #据说影响很大
     # 'alpha': 0.4,
     'silent': 1,
     'n_estimators': 1600
    } 


cat_params = {'learning_rate': 0.03,
              'iterations': 2000,
              'loss_function': 'Logloss', 
              'eval_metric': 'AUC',
              'depth': 7,
              'l2_leaf_reg': 5,
              'random_seed': 0,
              'thread_count': 40
              }

rf_params = {'n_estimators': 700,
             'n_jobs': 20,
            }

adb_params = {'n_estimators':100
            }

etc_params = {'n_estimators':1000,
              'n_jobs': 20,
            }

kfold = 10
lgb = LGBMClassifier(**lgb_params)
cat = CatBoostClassifier(**cat_params)
gbdt = GradientBoostingClassifier()
adb = AdaBoostClassifier(**adb_params)
rf = RandomForestClassifier(**rf_params)
lr = LogisticRegression(C=0.1)
knn = KNeighborsClassifier(n_jobs=-1)
mlp = MLPClassifier()
gn = BernoulliNB()
xgb = XGBClassifier(**xgb_params)
etc = ExtraTreesClassifier(**etc_params)

# 目前就用了这几个，继续试其他的
base_models =({etc})
folds = pickle.load(open('folds.pkl','rb'))

S_train = np.zeros((train_X.shape[0], len(base_models)))
S_test = np.zeros((test_X.shape[0], len(base_models)))
for i, clf in enumerate(base_models):
    model = str(clf).split('(')[0]
    if len(model) > 40:
        model = str(clf).split('.')[2].split(' ')[0]
    print('Running {}'.format(model))
    X = train_X.copy()
    y = train_Y.copy()
    T = test_X.copy()
    if model not in ['LGBMClassifier','XGBClassifier','CatBoostClassifier']:
        X[np.isnan(X)] = 0
        T[np.isnan(T)] = 0
        ms = StandardScaler()
        XT = np.vstack((X,T))
        XT = ms.fit_transform(XT)
        X = XT[0:len(X)]
        T = XT[len(X):]
    skf = StratifiedKFold(n_splits=kfold, random_state=1)
    S_test_i = np.zeros((T.shape[0], kfold))
    cv = 0
    for j, (train_index, test_index) in enumerate(folds):
        X_train, X_eval = X[train_index], X[test_index]
        y_train, y_eval = y[train_index], y[test_index]
        if model in ['LGBMClassifier','XGBClassifier',]:
           # clf.set_params(random_state=j)
            clf.fit(X_train, y_train, eval_set=[(X_eval,y_eval)], early_stopping_rounds=100,eval_metric='auc',verbose=False)
        elif model == 'CatBoostClassifier':
            clf.fit(X_train, y_train, eval_set=[X_eval,y_eval], use_best_model=True,verbose=False)
        else:
            clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_eval)[:,1]
        S_train[test_index, i] = y_pred
        S_test_i[:, j] = clf.predict_proba(T)[:,1]
        fpr, tpr, thresholds = roc_curve(y_eval, y_pred)
        cv += auc(fpr, tpr) / kfold
    S_test[:, i] = S_test_i.mean(axis=1)
    print('Model socre is {}'.format(cv))
    
# 保存元特征
pickle.dump(S_train, open('xiong_train_etc.pkl', 'wb'))
pickle.dump(S_test, open('xiong_test_etc.pkl', 'wb'))

