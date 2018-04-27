from lightgbm import LGBMClassifier
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import auc,roc_curve,accuracy_score,roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.preprocessing import MinMaxScaler,StandardScaler,scale

trainset = pickle.load(open('data/xiong_trainset.pkl', 'rb'))
testset = pickle.load(open('data/xiong_testset.pkl', 'rb'))

trainset.gender=trainset.gender.apply(lambda x:1 if x=='男' else x)
trainset.gender=trainset.gender.apply(lambda x:0 if x=='女' else x)
trainset.age=trainset.age.apply(lambda x:0 if x=='00后' else x)
trainset.age=trainset.age.apply(lambda x:1 if x=='90后' else x)
trainset.age=trainset.age.apply(lambda x:2 if x=='80后' else x)
trainset.age=trainset.age.apply(lambda x:3 if x=='70后' else x)
trainset.age=trainset.age.apply(lambda x:4 if x=='60后' else x)

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

train_features = [x for x in trainset.columns if x not in ['label','userid','province']]

train_X = trainset[train_features].values
train_Y = trainset['label'].values
test_X = testset[train_features].values


lgb_params1 = {
'task': 'train',
'boosting_type': 'dart',
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
'n_estimators': 2200,
'is_unbanlance':'true'
}  
'''
lgb_params1 = {
'task': 'train',
'boosting_type': 'dart',
'objective': 'binary',
'metric': 'auc',
'metric_freq':1,
'is_training_metric':'false',
# 'max_bin':255,
# 'num_leaves': 180,
'learning_rate': 0.02,
'tree_learner':'serial',
'feature_fraction': 0.8,
'feature_fraction_seed':3,
'bagging_fraction': 0.8,
'bagging_freq': 5,
# 'min_date_in_leaf':100,
# 'min_sum_hessian_in_leaf':100,
'early_stopping_rounds':80,
'verbose':0,
'n_estimators': 2600,
'lambda_l1':0.5,
'lambda_l2':0.5
}
'''
'''
ext_params1 = {
'n_estimators':600,
    'n_jobs':20,}
'''
ext_params2 = {
'n_estimators':800,
    'n_jobs':20,}

kfold = 10
lgb1 = LGBMClassifier(**lgb_params1)
#ext1 = ExtraTreesClassifier(**ext_params1)
ext2 = ExtraTreesClassifier(**ext_params2)

#base_models = (ext1, ext2)
base_models = (lgb1, ext2)
folds = pickle.load(open('data/folds.pkl','rb'))

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
#     skf = StratifiedKFold(n_splits=kfold, random_state=1)
    S_test_i = np.zeros((T.shape[0], kfold))
    cv = 0
    for j, (train_index, test_index) in enumerate(folds):
        X_train, X_eval = X[train_index], X[test_index]
        y_train, y_eval = y[train_index], y[test_index]
        if model in ['LGBMClassifier','XGBClassifier',]:
            clf.set_params(random_state=j)
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

    pickle.dump(S_train, open('data/Strain_xiong.pkl', 'wb'))
    pickle.dump(S_test, open('data/Test_xiong.pkl', 'wb'))

#score of lgbdart1_2 is too low