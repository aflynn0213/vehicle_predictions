# -*- coding: utf-8 -*-
"""
Created on Tues Dec  5 15:17:49 2023

@author: lu516e
"""

from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, ElasticNet

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, 
                             r2_score, 
                             auc, 
                             roc_auc_score,
                             roc_curve,
                             confusion_matrix)

from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm

from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, 
                                   PolynomialFeatures, StandardScaler,
                                   label_binarize)

from skopt import BayesSearchCV
from skopt.space import Real,Categorical,Integer
#import warnings as wrn


def dropFeatures(df_,drop_outputs=False):
    #INCLUDE ALL INPUT FEATURES IN THE X DATAFRAME
    feats_to_drop = ['sellerispriv',
                     'sellername',
                     'sellerzip',
                     'vehbodystyle',
                     'vehmodel', # 1:1 CORRESPONDANCE WITH VehMake
                     'vehtype',
                     'vehtransmission']
    if drop_outputs:
        #OUTPUT LABELS
        output_columns = [col.lower() for col in getOutputFeatures()]
        feats_to_drop.extend(output_columns)
    
    return df_.drop(feats_to_drop,axis=1)
    

def preprocessInput(df_,train=False):
    #MAKE LISTINGID INDEX COLUMN
    temp = df_.copy()
    temp.set_index('ListingID',inplace=True)
    temp.columns = temp.columns.str.lower()
    temp = dropFeatures(temp,drop_outputs=train)
    temp,cat_cols,num_cols = arbitrateFeatuteTypes(temp)
    
    print("SCALING NUMERICAL VALUES.....")
    df_num = zScoreTransform(temp, num_cols)
    
    print("ENCODING......")
    df_cat = labelEncode(temp,cat_cols)
    
    drops = num_cols + cat_cols
    
    df_notes = tf_idfTokenizer(temp["vehsellernotes"])
    
    drops.append("vehsellernotes")
    temp.drop(drops,axis=1,inplace=True)
    
    #print(temp.columns)
    df_num = pd.DataFrame(df_num)
    df_cat = pd.DataFrame(df_cat)
    df_notes = pd.DataFrame(df_notes)
    #TODO
    #FEATS,vehhistory, vehsellernotes
    df_concat = pd.concat([df_num,df_cat,df_notes],axis=1)
    df_concat.index = df_.ListingID
    return df_concat

    
def getOutputFeatures():
    return ['Vehicle_Trim', 'Dealer_Listing_Price']


def arbitrateFeatuteTypes(df_):
    cat_feats = []
    num_feats = []
    for col in df_.columns:
        if df_[col].dtype == 'object':
            print("OBJECT COL: ",col)
            # Categorical column - fill missing values with 'Unknown'
            df_[col].fillna('Unknown', inplace=True)
            cat_feats.append(col)        
        else:
            print("Numerical col: ",col)
            # Numerical column - fill missing values with mean
            df_[col].fillna(df_[col].mode(), inplace=True)
            num_feats.append(col)
    return df_,cat_feats,num_feats



def labelEncode(df_,cat_cols):
    encoder = OneHotEncoder()
    return encoder.fit_transform(df_[cat_cols])

    
def zScoreTransform(df_,num_cols):
    scaler = StandardScaler()
    return scaler.fit_transform(df_[num_cols])
    
    
def tf_idfTokenizer(df_):
    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(df_)


if __name__ == "__main__":
    print("SETTING UP.....")
    #wrn.filterwarnings('ignore', category=SettingWithCopyWarning )
    df_train = pd.read_csv('Training_DataSet.csv')
    df_test = pd.read_csv('Test_Dataset.csv')
    
    df_train.dropna(axis=0,how='any',inplace=True)
    
    #SETUP TRAINING AND TEST INPUT DATAFRAMES
    x_train = preprocessInput(df_train,True)
    x_test = preprocessInput(df_test) 
    print(x_train)
    x_train.to_csv("xtr.csv")
    quit()
    
    #SET LABELS TO PREDICT IN Y DATAFRAME
    y_train = df_train[getOutputFeatures()]   
    trim = y_train.iloc[:,0]
    lab_encoder = LabelEncoder()
    trim = lab_encoder.fit_transform(trim)
    seller_price = y_train.iloc[:,1]
    
    # for oh in one_hot_cols:
    #     x_train[oh] = one_hot_encoder.fit_transform(x_train[[oh]]).toarray()
    #     x_test[oh] = one_hot_encoder.fit_transform(x_test[[oh]]).toarray()
    
    param_grid_knn = {
        'n_neighbors': [3, 5, 10,15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2],
        'n_jobs': [-1]
    }
    print("RUNNING REGRESSION....")
    knn = KNeighborsRegressor()
    knn_cv = GridSearchCV(knn, param_grid_knn,cv=10)
    knn_cv.fit(x_train,seller_price)
    pred_price = knn_cv.best_estimator_.predict(x_test)
    
    #bayes = BayesSearchCV()
    
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'n_jobs':[-1]
    }
    
    print("RUNNING CLASSIFICATION.....")
    rf_class = RandomForestClassifier()
    rf_cv = GridSearchCV(rf_class, param_grid_rf,cv=5)
    rf_cv.fit(x_train,trim)
    pred_trim = rf_cv.best_estimator_.predict(x_test)

    df_preds = x_test.copy()
    df_preds[output_columns[0]] = pred_trim
    df_preds[output_columns[1]] = pred_price
    df_preds.to_csv("Predictions.csv")
    
    