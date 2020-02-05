# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:40:51 2019

@author: Bryan
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import datetime
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from lightgbm import LGBMClassifier
import math


train_transaction = pd.read_csv(r'G:\Kaggle\ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv(r'G:\Kaggle\ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv(r'G:\Kaggle\ieee-fraud-detection/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv(r'G:\Kaggle\ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv(r'G:\Kaggle\ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)




##Missing values: drop columns that are missing more than 80 percent of the values
missingDataList = pd.DataFrame(train.isnull().sum(), columns=['numMiss'])
missingDataList1 = missingDataList["numMiss"].sort_values(ascending=False)
morethan80 = missingDataList['numMiss'].loc[missingDataList['numMiss'] > 405352]
#morethan80 = missingDataList['numMiss'].loc[missingDataList['numMiss'] > 400000]
morethan80 = list(morethan80.index)
"""
I will list some features to leave here for feature engineering, but these should be dropped later before fitting the model.
D6, D7, D8, D9, D12, D13, D14, DeviceType, DeviceInfo
"""
save_features = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14', 'DeviceType', 'DeviceInfo', 'M4']

morethan80_9 = [feature for feature in morethan80 if feature not in save_features]

train.drop(morethan80_9, axis=1, inplace=True)
test.drop(morethan80_9, axis=1, inplace=True)


#Try down sampling#####################
train['isFraud'].value_counts()
# Separate majority and minority classes
df_majority = train[train.isFraud==0]
df_minority = train[train.isFraud==1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=165304,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
df_downsampled.isFraud.value_counts()
#End downsampling######################

#Put the downsampled data back into train
train = df_downsampled


#Create the target variable
y_train = train['isFraud'].copy()

# Drop target, fill in NaNs
X_train = train.drop('isFraud', axis=1)
X_test = test.copy()


X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

del train, test, train_transaction, train_identity, test_transaction, test_identity, df_downsampled, df_majority, df_majority_downsampled, df_minority

#Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))   
    
    
def bucketingFeatures(df):
    
    df["bucket_D1"] = pd.cut(df['D1'], 5, labels=range(1, 6))
    df["bucket_D1"] = pd.to_numeric(df["bucket_D1"])
    df.drop('D1', axis=1, inplace=True)
    
    df["bucket_D2"] = pd.cut(df['D2'], 5, labels=range(1, 6))
    df["bucket_D2"] = pd.to_numeric(df["bucket_D2"])
    df.drop('D2', axis=1, inplace=True)
    
    df["bucket_D3"] = pd.cut(df['D3'], 6, labels=range(1, 7))
    df["bucket_D3"] = pd.to_numeric(df["bucket_D3"])
    df.drop('D3', axis=1, inplace=True)
    
    df["bucket_D4"] = pd.cut(df['D4'], 6, labels=range(1, 7))
    df["bucket_D4"] = pd.to_numeric(df["bucket_D4"])
    df.drop('D4', axis=1, inplace=True)
    
    df["bucket_D5"] = pd.cut(df['D5'], 6, labels=range(1, 7))
    df["bucket_D5"] = pd.to_numeric(df["bucket_D5"])
    df.drop('D5', axis=1, inplace=True)
    
    
    df["bucket_D6"] = pd.cut(df['D6'], 10, labels=range(1, 11))
    df["bucket_D6"] = pd.to_numeric(df["bucket_D6"])
    df.drop('D6', axis=1, inplace=True)
    
    df["bucket_D7"] = pd.cut(df['D7'], 10, labels=range(1, 11))
    df["bucket_D7"] = pd.to_numeric(df["bucket_D7"])
    df.drop('D7', axis=1, inplace=True)
    
    df["bucket_D8"] = pd.cut(df['D8'], 10, labels=range(1, 11))
    df["bucket_D8"] = pd.to_numeric(df["bucket_D8"])
    df.drop('D8', axis=1, inplace=True)
    
    df["bucket_D9"] = pd.cut(df['D9'], 10, labels=range(1, 11))
    df["bucket_D9"] = pd.to_numeric(df["bucket_D9"])
    df.drop('D9', axis=1, inplace=True)
    
    df["bucket_D12"] = pd.cut(df['D12'], 10, labels=range(1, 11))
    df["bucket_D12"] = pd.to_numeric(df["bucket_D12"])
    df.drop('D12', axis=1, inplace=True)
    
    df["bucket_D13"] = pd.cut(df['D13'], 10, labels=range(1, 11))
    df["bucket_D13"] = pd.to_numeric(df["bucket_D13"])
    df.drop('D13', axis=1, inplace=True)
    
    df["bucket_D14"] = pd.cut(df['D14'], 10, labels=range(1, 11))
    df["bucket_D14"] = pd.to_numeric(df["bucket_D14"])
    df.drop('D14', axis=1, inplace=True)
    
    df["bucket_DeviceType"] = pd.cut(df["DeviceType"], 3, labels=range(1, 4))
    df["bucket_DeviceType"] = pd.to_numeric(df["DeviceType"])
    df.drop('DeviceType', axis=1, inplace=True)
    
    df["bucket_DeviceInfo"] = pd.cut(df['DeviceInfo'], 10, labels=range(1, 11))
    df["bucket_DeviceInfo"] = pd.to_numeric(df["DeviceInfo"])
    df.drop('DeviceInfo', axis=1, inplace=True)
    
  
bucketingFeatures(X_train)
bucketingFeatures(X_test)


def featureEngineering(df):
    df['groupbyProduct'] = df.groupby(['ProductCD'])['TransactionAmt'].transform('mean')
    df['groupbyM4'] = df.groupby(['M4'])['TransactionAmt'].transform('mean')
    df.drop('ProductCD', axis=1, inplace=True)
    df.drop('M4', axis=1, inplace=True)

featureEngineering(X_train)   
featureEngineering(X_test)


X_train_hold, X_test_hold, y_train_hold, y_test_hold = train_test_split(X_train, y_train, test_size=0.2, random_state=123, stratify=y_train)

#Filter feature selection method
X_train_hold['isFraud'] = y_train_hold
cor = X_train_hold.corr()
#Correlation with output variable
cor_target = abs(cor["isFraud"])
#Selecting highly correlated features
irrelevant_features = cor_target[cor_target < 0.005]
irrelevant_features = list(irrelevant_features.index)
df = cor["isFraud"].sort_values(ascending=False)
#End filter feature selection method
X_train_hold.drop('isFraud', axis=1, inplace=True)

for item in irrelevant_features:
    X_train.drop(item, axis=1, inplace=True)
    X_test.drop(item, axis=1, inplace=True)
    

#Model
clf2 = xgb.XGBClassifier(n_estimators=500,
                        eval_metric="auc",
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        missing=-999,
                        random_state=123)

clf2.fit(X_train_hold, y_train_hold)

y_predicted = clf2.predict_proba(X_test_hold)[:,1]

roc_auc_score(y_test_hold,y_predicted)




clf = xgb.XGBClassifier(n_estimators=500,
                        eval_metric="auc",
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        missing=-999,
                        random_state=123)

clf.fit(X_train, y_train)

sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]

sample_submission.to_csv(r'G:\Kaggle/irreldropped_featureEngineering.csv')

