#!/usr/bin/env python
# coding: utf-8

# In[1773]:


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm

from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, 
                                   PolynomialFeatures, StandardScaler,
                                   label_binarize,MultiLabelBinarizer)
 
from scipy.sparse import csr_matrix

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from skopt import gp_minimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.base import BaseEstimator, RegressorMixin

from importlib import reload
    
import trainers as tr
import price_calculations as pr
import neural_models as nm
import xgboost as xgb

import matplotlib.pyplot as plt


# In[1774]:


# Reload Trainers due to possibility of local changes
reload(tr)
reload(pr)
reload(nm)


# In[1775]:


def engineerTestData(df,log_cols,encoded_cols,freq_cols,
                     mask_cols,token_cols,orig_cols, feats_to_drop,
                     coder,token_1,token_2):
    df.columns =  df.columns.str.lower()
    df.columns = [col.strip() for col in df.columns]
    df.set_index('listingid',inplace=True)
    df = df.apply(lambda col: col.fillna('Unknown') if col.dtype == 'O' else col.fillna(0))
    
    [df.__setitem__(col, np.log(np.ceil(df[col]))) for col in log_cols]
    [df.__setitem__(col, df[col].map(df[col].value_counts())) for col in freq_cols]
    [df.__setitem__(col, df[col].astype(int)) for col in mask_cols]

    handle_encode = orig_cols 
    test_encode = df[encoded_cols]
    for col in handle_encode:
        col = col.strip()   
        func_name = 'handle_' + col  # Prepare function name
        if func_name in globals() and callable(globals()[func_name]):
            func = globals()[func_name]
            if col == 'vehdrivetrain':
                temp_df = func(df[col].copy())  # Call the function dynamically
                test_encode[col] = temp_df
            elif col == 'vehhistory':
                df.loc[df[col] == 'Unknown', col] = 0
                temp_df = df[col].copy().str.split(',',n=1,expand=True)
                temp_df.columns = ['Owners', 'History']
                temp_df['Owners'] = temp_df['Owners'].str.extract(r'^(\d+)')
                encoded_hist = func(temp_df['History'])  # Call the function dynamically
                df['owners'] = temp_df['Owners']
                df[encoded_hist.columns] = encoded_hist
            elif col == 'vehengine':
                temp_df = func(df[col].copy())  # Call the function dynamically
                temp_df.columns = temp_df.columns.str.lower()
                df[temp_df.columns] = temp_df
            elif col == 'vehcolorext':
                col_temp = func(df[col].copy())  # Call the function dynamically
                col_temp.columns = col_temp.columns.str.lower()
            elif col == 'vehcolorint':
                col_tmp = func(df[col].copy())  # Call the function dynamically
                col_tmp.columns = col_tmp.columns.str.lower()            
        else:
            print(f"Function '{func_name}' does not exist or is not callable.")

    colors = pd.merge(col_temp,col_tmp,left_index=True, right_index=True)
    df = pd.merge(df,colors,left_index=True, right_index=True)
    temp_encoded = oHotEncode(test_encode,coder)
    df.drop(columns=encoded_cols,inplace=True)
    df = pd.merge(df,temp_encoded,left_index=True, right_index=True)
    df.columns = df.columns.astype(str)

    tf1 = tf_idfTokenizer(df[token_cols[0]].copy(),token_1)
    tf2 = tf_idfTokenizer(df[token_cols[1]].copy(),token_2)
    #tfs = pd.concat([tf1, tf2])
    #tfs = combined_tf.loc[:,~combined_tf.columns.duplicated()] 
    tfs = pd.merge(tf1, tf2,left_index=True,right_index=True)
    df = pd.merge(df,tfs,left_index=True,right_index=True)
    df.drop(columns=feats_to_drop,inplace=True)
    
    return df
    
def oHotEncode(df_,coder):
    encoded_mat = coder.transform(df_)
    return pd.DataFrame(encoded_mat.todense(),
                        columns=[cat for columns in coder.categories_ for cat in columns],
                        index=df_.index)

    
def zScoreTransform(col):
    return np.divide(np.subtract(col,col.mean()),col.std())

    
def tf_idfTokenizer(df_,tfidf):
    tf_mat = tfidf.transform(df_)
    return pd.DataFrame(tf_mat.toarray(),
                          columns=tfidf.get_feature_names_out(['feature']),
                          index=df_.index)

    
def setFeatPtr(data,index):
    return data.iloc[:,index],data.columns[index]


def plotDist(data,title):
    # Plotting a histogram of frequencies
    fig, ax = plt.subplots()
    sns.histplot(data, kde=True, ax=ax)
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    plt.show()

def categorize_train(phrase, awd_pattern, fwd_pattern, wd_pattern):
    if re.search(awd_pattern, phrase) and re.search(wd_pattern, phrase):
        return 'hybrid'
    elif re.search(awd_pattern, phrase):
        return 'awd'
    elif re.search(fwd_pattern, phrase):
        return 'fwd'
    elif re.search(wd_pattern, phrase):
        return '_4_wd'
    else:
        return 'Unknown'

def handle_vehdrivetrain(df):
    df = df.str.lower()
    awd_pattern = re.compile(r'awd|all', flags=re.IGNORECASE)
    fwd_pattern = re.compile(r'fwd|front', flags=re.IGNORECASE)
    wd_pattern = re.compile(r'4x4|4wd|four\s?WHEEL\s?DRIVE\b', flags=re.IGNORECASE)
    return df.apply(categorize_train, args=(awd_pattern, fwd_pattern, wd_pattern))

# Function extracts engine size and configuration
def categorize_engine(phrase):
    engine_size_match = re.search(r'\b\d+(\.\d+)?\s*L\b', phrase)  # Matches pattern with number (with or without decimal) followed by L
    config_match = re.search(r'V[-]?6|V[-]?8|\b\d\s*cylinder|\b6\s*cylinde', phrase, re.IGNORECASE)  # Matches V6, V-6, V8, V-8, or a number followed by cylinders

    if engine_size_match:
        engine_size = float(re.search(r'\d+(\.\d+)?', engine_size_match.group()).group())  # Extracts engine size
        size_category = engine_size  # Assigning the engine size directly as the size category
    else:
        size_category = 0

    if config_match:
        config_str = config_match.group().upper()
        config = 6 if '6' in config_str else 8  # Assign 6 or 8 based on the presence of 'Vx' or 'Cyclinders'
    else:
        config = 0

    return size_category, config

def handle_vehengine(df):
    extracted_info = df.apply(categorize_engine)
    # Convert the extracted information into a DataFrame
    df = pd.DataFrame(extracted_info.tolist(), columns=['EngineSize', 'Cylinders'], index=df.index)
    return df

def handle_vehhistory(df):
    print("HISTORY")
    # List of unique phrases
    unique_phrases = [
        'Accident(s) Reported',
        'Buyback Protection Eligible',
        'Non-Personal Use Reported',
        'Title Issue(s) Reported'
    ]
    
    # Strip whitespace from each string in the Series
    df = df.astype(str).str.strip()
    
    # Initialize a DataFrame to store the encoded values
    encoded_df = pd.DataFrame(index=df.index)
    
    # Iterate over each unique phrase
    for phrase in unique_phrases:
        # Check if the phrase exists in each row and create a binary indicator
        encoded_df[phrase] = df.apply(lambda x: 1 if phrase in x else 0)
    
    # Create a 'None of the above' column to indicate if none of the phrases were found
    encoded_df['None of the above'] = (encoded_df.sum(axis=1) == 0).astype(int)
    
    return encoded_df

def handle_vehcolorext(df_):
    print("COLOR")
    common_colors = ['Black', 'Blue', 'Brown', 'Gray', 'Green', 'Steel', 'Metallic','Pearlcoat', 'Clearcoat',
                     'Charcoal','Granite', 'Red', 'Silver', 'White']
    silver_colors = ['Gray', 'Steel', 'Charcoal', 'Silver']
    
    temp = pd.DataFrame(index=df_.index)
    for color in common_colors:
        temp[f'{color}'] = df_.str.contains(color, case=False).astype(int)

    # Grouping similar silver colors into a single 'Silver' category
    temp['Silver'] = df_.str.contains('|'.join(silver_colors), case=False).astype(int)
    temp.drop([col for col in silver_colors if col != 'Silver'], axis=1, inplace=True)
    
    # Populates a 'None' category if none of the common colors are present
    temp['None'] = 1 - temp[[f'{color}' for color in temp.columns]].max(axis=1)
    
    return temp

def handle_vehcolorint(df_):
    print("COLOR2")
    common_colors = ['Black', 'Blue', 'Brown', 'Gray', 'Steel', 'Beige','trim',
                     'Charcoal','Red', 'Silver', 'Frost','Maple','Tan','Cirrus','carbon','plum']
    silver_colors = ['Gray', 'Steel', 'Charcoal', 'Silver']
    temp = pd.DataFrame(index=df_.index)
    for color in common_colors:
        temp[f'{color}'] = df_.str.contains(color, case=False).astype(int)

    # Grouping similar silver colors into a single 'Silver' category
    temp['Silver'] = df_.str.contains('|'.join(silver_colors), case=False).astype(int)
    temp.drop([col for col in silver_colors if col != 'Silver'], axis=1, inplace=True)
    
    # Populates a 'None' category if none of the common colors are present
    temp['None'] = 1 - temp[[f'{color}' for color in temp.columns]].max(axis=1)
    
    return temp

def calculate_age(df_):
    age = 2024 - df_
    return age


# In[1776]:


#Initialize training and test dataframes
orig_train = pd.read_csv('Training_DataSet.csv')
df_test = pd.read_csv('Test_Dataset.csv')


#Drop blank cells from training set to clean up data (contemplated using mean, median, or mode imputation,
#but will explore without corrupting the data and due to the large size of the dataset eliminating
#some rows should suffice
orig_train.dropna(axis=0,how='any',inplace=True) #EXPLICIT CALL TO DROP ROWS WITH A SINGLE MISSING VALUE
                                                 #(DEFAULT CALL DOES SAME)


# In[1777]:


orig_train.columns = orig_train.columns.str.lower()
orig_train.set_index('listingid',inplace=True)

df_train = orig_train.copy()
df_train.info()


# In[1778]:


#NOTICE THERES ONLY JEEPS AND CADILLACS IN DATA SET BRAKE THEM UP FURTHER TO SEE
#THE TRIMS SINCE TRIMS ARE USUALLY EXCLUSIVE TO MANUFACTURER LINE
jeeps = df_train[df_train['vehmake'].str.lower() == 'jeep'].copy()
caddy = df_train[df_train['vehmake'].str.lower() == 'cadillac'].copy()

print(jeeps['vehicle_trim'].value_counts())
print(caddy['vehicle_trim'].value_counts())


# In[1779]:


#MASSIVE CLASS IMBALANCE WILL NEED TO CONDENSE THIS AND IGNORE LOW FREQUENCY CLASSES
#BECAUSE THEY ADD NOISE AND CLASSIFIER WILL NOT BE ABLE TO ARBITRATE
conditions = [
    caddy['vehicle_trim'].str.lower().str.contains('premium'),
    caddy['vehicle_trim'].str.lower().str.contains('luxury'),
    caddy['vehicle_trim'].str.lower().str.contains('base'),
    caddy['vehicle_trim'].str.lower().str.contains('platinum')
]

choices = ['Premium Luxury', 'Luxury', 'Base', 'Platinum']

# Use np.select() to relabel based on conditions
caddy['vehicle_trim'] = np.select(conditions, choices, default='other')

# Filter the DataFrame to keep only rows labeled as 'premium', 'luxury', 'base', or 'platinum'
valid_labels = ['Premium Luxury', 'Luxury', 'Base', 'Platinum']
caddy = caddy[caddy['vehicle_trim'].isin(valid_labels)]
caddy["vehicle_trim"]


# In[1780]:


valid_labels_jeep = ['limited', 'laredo',  'summit',
                     'overland', 'altitude','trailhawk',
                     'trackhawk','srt','sterling']

conditions_jeep = [
    jeeps['vehicle_trim'].str.lower().str.contains(label) for label in valid_labels_jeep
]

choices_jeep = ['Limited', 'Laredo',  'Summit', 
                     'Overland', 'Altitude','Trailhawk', 'Trackhawk',
                        'SRT','Sterling Edition']

# Use np.select() to classify based on conditions
jeeps['vehicle_trim'] = np.select(conditions_jeep, choices_jeep, default='other')
print(jeeps['vehicle_trim'].value_counts())
# Filter the DataFrame to keep only rows labeled with valid labels
jeeps = jeeps[jeeps['vehicle_trim'].isin(choices_jeep)]
jeeps["vehicle_trim"]


# In[1781]:


print("CADDY")
print(caddy["vehicle_trim"].value_counts())
print("JEEP")
print(jeeps["vehicle_trim"].value_counts())


# In[1782]:


print(jeeps.index)
print(caddy.index)
df_train.update(jeeps[['vehicle_trim']])
df_train.update(caddy[['vehicle_trim']])
df_train["vehicle_trim"].value_counts()


# In[1783]:


options = choices + choices_jeep
df_train = df_train[df_train['vehicle_trim'].isin(options)]

print(df_train["vehicle_trim"].value_counts())
df_train.head()


# In[1784]:


feats_to_drop = []
encoded_cols = []
freq_cols = []
same_cols = []
mask_cols = []
log_cols = [] 
orig_cols = []


jeeps = df_train[orig_train["vehmake"]=="Jeep"].copy()
caddys = df_train[df_train["vehmake"]=="Cadillac"]

input_jeeps = jeeps.copy()
input_jeeps = input_jeeps.iloc[:,:-2]

input_caddys = caddys.copy()
input_caddys = input_caddys.iloc[:,:-2]

col = 0
feat_ptrj,column = setFeatPtr(input_jeeps,col)
feat_ptrc,column = setFeatPtr(input_caddys,col)


# In[1785]:


feat_ptrj.head()


# In[1786]:


feat_ptrc.head()


# In[1787]:


#PERCENTAGE MODE APPEARS
count = (feat_ptrj==feat_ptrj.mode()[0]).sum()
print(count/len(feat_ptrj))
print(feat_ptrj.nunique())
count = (feat_ptrc==feat_ptrc.mode()[0]).sum()
print(count/len(feat_ptrc))
print(feat_ptrc.nunique())


# In[1788]:


value_counts = feat_ptrj.value_counts()
# Plotting a histogram of frequencies (Frequencies of Frequencies)
plotDist(value_counts,"Density of value frequencies")
#FREQUENCY ENCODE THESE VALUES AND THEN TAKE Z SCORE OR THE FREQUENCIES
zvalues = zScoreTransform(value_counts)
print(zvalues)
plotDist(zvalues,"Density of Z-transformed frequencies")
# Assuming 'value_counts' contains the frequencies
log_frequencies = np.log(value_counts)
# Plotting the density plot of the log of frequencies
plt.figure(figsize=(8, 6))
plotDist(log_frequencies,"Density of log of frequencies")
# Plotting the density plot of the log of frequencies
zlog = zScoreTransform(log_frequencies)
plotDist(zlog,"Density of Z-transformed log of frequencies")
freq = feat_ptrj.value_counts().to_dict()
feat_ptrj = feat_ptrj.map(freq)
feat_ptrj.head()


# In[1789]:


value_counts = feat_ptrc.value_counts()
# Plotting a histogram of frequencies (Frequencies of Frequencies)
plotDist(value_counts,"Density of value frequencies")
#FREQUENCY ENCODE THESE VALUES AND THEN TAKE Z SCORE OR THE FREQUENCIES
zvalues = zScoreTransform(value_counts)
print(zvalues)
plotDist(zvalues,"Density of Z-transformed frequencies")
# Assuming 'value_counts' contains the frequencies
log_frequencies = np.log(value_counts)
# Plotting the density plot of the log of frequencies
plt.figure(figsize=(8, 6))
plotDist(log_frequencies,"Density of log of frequencies")
# Plotting the density plot of the log of frequencies
zlog = zScoreTransform(log_frequencies)
plotDist(zlog,"Density of Z-transformed log of frequencies")
freq = feat_ptrc.value_counts().to_dict()
feat_ptrc = feat_ptrc.map(freq)
feat_ptrc.head()


# In[1790]:


input_jeeps[column] = feat_ptrj
input_jeeps.head()


# In[1791]:


input_caddys[column] = feat_ptrc
input_caddys.head()


# In[1792]:


freq_cols.append(column)
col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrj)
print(feat_ptrc)


# In[1793]:


print(feat_ptrj.value_counts())
print(feat_ptrc.value_counts())


# In[1794]:


feats_to_drop.append(column)

col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj.nunique())
print(feat_ptrj.unique())

feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc.nunique())
print(feat_ptrc.unique())


# In[1795]:


#A CATEGORY COLUMN EASY TO ONE HOT ENCODE WITH A SMALL ENUMERATION AMOUNT (ONLY REQUIRES
# 5 COLUMNS TO ENCODE)
encoded_cols.append(column)
print(feat_ptrj)
print(feat_ptrc)

col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj)
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc)


# In[1796]:


print(feat_ptrj.value_counts()[feat_ptrj.mode()]/len(feat_ptrj))
feat_ptrj.value_counts().head(30)


# In[1797]:


#NOT CATEGORICAL OR CONTAINS DOMINATE VALUES, WILL NOT SIGNFICANTLY IMPACT MODEL PREDICTION EFFICIENCY
feats_to_drop.append(column)


# In[1798]:


col +=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj)
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc)


# In[1799]:


#POSSIBLY NORMALIZE (Z-TRANSFORM) FOR NOW KEEP IT INTACT
same_cols.append(column)
col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj)
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc)


# In[1800]:


plotDist(feat_ptrj,'Density of Seller Review Count')
plotDist(np.log(feat_ptrj),'Density of Log(Seller Review Count)')
plotDist(zScoreTransform(feat_ptrj),'Density of Z-Transform of Seller Review Count')
plotDist(zScoreTransform(np.log(feat_ptrj)),'Density of Z-Transform of Log of Seller Review Count')


# In[1801]:


#KEEP REVIEW COUNT AS IS FOR NOW
same_cols.append(column)

col+=1 
feat_ptrj,column = setFeatPtr(input_jeeps,col)
feat_ptrj
feat_ptrc,column = setFeatPtr(input_caddys,col)
feat_ptrc


# In[1802]:


#STATES -> CATEGORICAL
encoded_cols.append(column)

col+=1 
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj.nunique())
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc.nunique())


# In[1803]:


#ZIP SEEMS REDUNDANT WITH CITY/STATE INFO ALREADY EXISTING
#PLUS THE AMOUNT OF VARYING ZIPS PROVIDES NOISY DATA
feats_to_drop.append(column)
print(feats_to_drop)

col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj)
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc)


# In[1804]:


print(feat_ptrj.nunique())
print(feat_ptrc.nunique())


# In[1805]:


#ALL SUV, MEANINGLESS DATA
feats_to_drop.append(column)
col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj)
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc)


# In[1806]:


#MASK BOOLEANS AS 1 AND 0's
feat_ptrj = (feat_ptrj).astype(int)
input_jeeps[column] = feat_ptrj
feat_ptrc = (feat_ptrc).astype(int)
input_caddys[column] = feat_ptrc
mask_cols.append(column)

col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj)
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc)


# In[1807]:


print(feat_ptrj.value_counts())
print(feat_ptrc.value_counts())


# In[1808]:


temp_dfj = handle_vehcolorext(feat_ptrj)
temp_dfj.columns = temp_dfj.columns.str.lower()
temp_dfc = handle_vehcolorext(feat_ptrc)
temp_dfc.columns = temp_dfc.columns.str.lower()

#Encoded with hand-written function rather than the encoder
self_encodej = pd.DataFrame(temp_dfj, index=temp_dfj.index,columns=temp_dfj.columns)
self_encodec = pd.DataFrame(temp_dfc, index=temp_dfc.index,columns=temp_dfc.columns)

orig_cols.append(column)
#Want to drop original
feats_to_drop.append(column)

print(temp_dfj.sum())
print(temp_dfj[temp_dfj["none"]==1].index)

print(temp_dfc.sum())
print(temp_dfc[temp_dfc["none"]==1].index)


# In[1809]:


col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj.value_counts())
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc.value_counts())


# In[1810]:


temp_dfj = handle_vehcolorint(feat_ptrj)
temp_dfj.columns = temp_dfj.columns.str.lower()
temp_dfc = handle_vehcolorint(feat_ptrc)
temp_dfc.columns = temp_dfc.columns.str.lower()

#Merge two handwritten encoded columns
self_encodej = pd.merge(self_encodej, temp_dfj, left_index=True,right_index=True)
self_encodec = pd.merge(self_encodec, temp_dfc, left_index=True,right_index=True)

orig_cols.append(column)
#Want to drop original
feats_to_drop.append(column)

print(temp_dfj.sum())
print(temp_dfj[temp_dfj["none"]==1].index)

print(temp_dfc.sum())
print(temp_dfc[temp_dfc["none"]==1].index)


# In[1811]:


col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj)
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc)


# In[1812]:


print(feat_ptrj.value_counts())
print(feat_ptrc.value_counts())


# In[1813]:


#BASED OFF UNIQUE VALUES SEPERATE INTO 4WD,FWD,or AWD
temp_dfj = handle_vehdrivetrain(feat_ptrj)
print(temp_dfj.value_counts())
temp_dfc = handle_vehdrivetrain(feat_ptrc)
print(temp_dfc.value_counts())


# In[1814]:


input_jeeps[column] = temp_dfj
input_caddys[column] = temp_dfc
encoded_cols.append(column)
orig_cols.append(column)
col+=1
print(encoded_cols)
print(input_jeeps[column])
print(input_caddys[column])


# In[1815]:


feat_ptrj,column = setFeatPtr(input_jeeps,col)
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrj.value_counts())
print(feat_ptrc.value_counts())


# In[1816]:


#handle_vehengine takes the vehEngine column and turns it into a 
#2 column data frame by splitting the phrases into engine size 
#and cyclinder configuration
temp_dfj = handle_vehengine(feat_ptrj)
temp_dfc = handle_vehengine(feat_ptrc)

print(temp_dfj["EngineSize"].value_counts())
print(temp_dfj["Cylinders"].value_counts())
print(temp_dfc["EngineSize"].value_counts())
print(temp_dfc["Cylinders"].value_counts())

# '0' represents unknown for either columns


# In[1817]:


input_jeeps[temp_dfj.columns] = temp_dfj
input_caddys[temp_dfc.columns] = temp_dfc

orig_cols.append(column)
feats_to_drop.append(column)

col+=1
print(encoded_cols)
print(temp_dfj)
print(temp_dfc)


# In[1818]:


feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj)
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc)


# In[1819]:


#ELIMINATE WORDS THAT APPEAR IN MORE THAN max_doc_freq OF DOCUMENTS (DOCUMENT ~ ROW)
#WILL GET RID OF COMMON WORDS SUCH AS "THE", "A", etc.
#LIMIT VOCABULARY TO max_feats COLUMNS (ONE FOR EACH WORD)
tf_featsj = TfidfVectorizer(max_df=0.50,max_features=30)
temp_dfj = feat_ptrj.copy()
tf_featsj = tf_featsj.fit(temp_dfj)
vocab1j = tf_idfTokenizer(temp_dfj,tf_featsj)
#THOUGHT: TUNE THE HYPERPARAMETERS TO OPTIMIZE THE TOKENIZER?
vocab1j.head()



# In[1820]:


#ELIMINATE WORDS THAT APPEAR IN MORE THAN max_doc_freq OF DOCUMENTS (DOCUMENT ~ ROW)
#WILL GET RID OF COMMON WORDS SUCH AS "THE", "A", etc.
#LIMIT VOCABULARY TO max_feats COLUMNS (ONE FOR EACH WORD)
tf_featsc = TfidfVectorizer(max_df=0.50,max_features=30)
temp_dfc = feat_ptrc.copy()
tf_featsc = tf_featsc.fit(temp_dfc)
vocab1c = tf_idfTokenizer(temp_dfc,tf_featsc)
#THOUGHT: TUNE THE HYPERPARAMETERS TO OPTIMIZE THE TOKENIZER?
vocab1c.head()


# In[1821]:


#DROP ORIGINAL STATE COLUMN AND LATER REPLACE WITH ENCODED MATRIX COLUMNS
feats_to_drop.append(column)
tokenize_cols = [column]
input_jeeps.head()


# In[1822]:


input_caddys.head()


# In[1823]:


col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj.value_counts())
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc.value_counts())


# In[1824]:


encoded_cols.append(column)

col+=1


# In[1825]:


feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj)
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc)

temp_dfj = feat_ptrj.str.split(',',n=1,expand=True)
temp_dfj.columns = ['Owners', 'History']
temp_dfj["History"].unique()


# In[1826]:


temp_dfc = feat_ptrc.str.split(',',n=1,expand=True)
temp_dfc.columns = ['Owners', 'History']
temp_dfc["History"].unique()


# In[1827]:


temp_dfj['Owners'] = temp_dfj['Owners'].str.extract(r'^(\d+)')
temp_dfc['Owners'] = temp_dfc['Owners'].str.extract(r'^(\d+)')

temp_dfj['Owners'].head()


# In[1828]:


input_jeeps['Owners'] = temp_dfj['Owners']
input_caddys['Owners'] = temp_dfc['Owners']

print(input_jeeps['Owners'])
print(input_caddys['Owners'])


# In[1829]:


temp_dfj["History"].value_counts()


# In[1830]:


#TURNS OUT THAT THESE PHRASES CAN ACTUALLY BE TURNED INTO CATEGORICAL COLUMNS
#EACH ELEMENT IS A COMBINATION OF VARYING SIZE OF THE 4 POSSIBLE UNIQUE PHRASES
#ONE HOT ENCODE WITH A COLUMN FOR EACH PHRASE
encoded_histj = handle_vehhistory(temp_dfj["History"])
encoded_histj.head()


# In[1831]:


encoded_histc = handle_vehhistory(temp_dfc["History"])
encoded_histc.head()


# In[1832]:


#DROP ORIGINAL COLUMN AND LATER REPLACE WITH ENCODED MATRIX COLUMNS
feats_to_drop.append(column)
self_encodej = pd.merge(self_encodej, encoded_histj, left_index=True, right_index=True)
self_encodec = pd.merge(self_encodec, encoded_histc, left_index=True, right_index=True)

orig_cols.append(column)
self_encodej.head()


# In[1833]:


col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj.value_counts())
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc.value_counts())
feat_ptrj.head()


# In[1834]:


#Use ceiling in order to round to whole days and start the listings 
#on day 1 rather than day 0
feat_ptrj = pd.Series(np.ceil(feat_ptrj),index=feat_ptrj.index)
feat_ptrc = pd.Series(np.ceil(feat_ptrc),index=feat_ptrc.index)

feat_ptrj.head()


# In[1835]:


plotDist(feat_ptrj,"Distribution of Listing Days Frequency")
plotDist(np.log(feat_ptrj),"Distribution of Log(Listing Days) Frequency")
plotDist(zScoreTransform(feat_ptrj),"Distribution of Z-Transform(Listing Days) Frequency")
plotDist(zScoreTransform(np.log(feat_ptrj)),"Distribution of Z-Tranform(Log(Listing Days)) Frequency")

plotDist(feat_ptrc,"Distribution of Listing Days Frequency")
plotDist(np.log(feat_ptrc),"Distribution of Log(Listing Days) Frequency")
plotDist(zScoreTransform(feat_ptrc),"Distribution of Z-Transform(Listing Days) Frequency")
plotDist(zScoreTransform(np.log(feat_ptrc)),"Distribution of Z-Tranform(Log(Listing Days)) Frequency")


# In[1836]:


#CHOOSE LOG VALUE
#feat_ptrj = np.log(feat_ptrj)
#feat_ptrc = np.log(feat_ptrc)

print(feat_ptrj)
print(feat_ptrc)


# In[1837]:


input_jeeps[column] = feat_ptrj
input_caddys[column] = feat_ptrc
log_cols.append(column)

col+=1
input_jeeps.head()


# In[1838]:


feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj.value_counts())
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc.value_counts())
feat_ptrj.head()


# In[1839]:


#The defining attribute of each list, going to keep the same for now in case the handler functions become "make" specific
same_cols.append(column)

col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj.value_counts())
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc.value_counts())


# In[1840]:


plotDist(feat_ptrj,"Density of Vehicle Mileage")
plotDist(np.log(feat_ptrj),"Density of Log(Vehicle Mileage)")
plotDist(zScoreTransform(feat_ptrj),"Density of Z-Transform(Vehicle Mileage)")
plotDist(zScoreTransform(np.log(feat_ptrj)),"Density of Z-Tranform(Log(Vehicle Mileage))")


# In[1841]:


#ORIGINAL DATA LOOKS ~NORMAL~

same_cols.append(column)
col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj.value_counts())
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc.value_counts())


# In[1842]:


#ALREADY HAVE JEEP/CADILLAC ENCODED COLUMNS WHICH HAVE A DIRECT CORRELATION TO THIS
#WILL REMOVE THIS EXTRANEOUS COLUMN
feats_to_drop.append(column)

col+=1 
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj.value_counts())
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc.value_counts())


# In[1843]:


encoded_cols.append(column)

col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
feat_ptrc,column = setFeatPtr(input_caddys,col)
feat_ptrj.head()


# In[1844]:


#ELIMINATE WORDS THAT APPEAR IN MORE THAN max_doc_freq OF DOCUMENTS (DOCUMENT ~ ROW)
#WILL GET RID OF COMMON WORDS SUCH AS "THE", "A", etc.
#LIMIT VOCABULARY TO max_feats COLUMNS (ONE FOR EACH WORD)
tfidfj = TfidfVectorizer(max_df=.50,max_features=60)
tf_revj = tfidfj.fit(feat_ptrj.copy())
vocab2j = tf_idfTokenizer(feat_ptrj,tf_revj)
#THOUGHT: TUNE THE HYPERPARAMETERS TO OPTIMIZE THE TOKENIZER?
vocab2j.head()


# In[1845]:


#ELIMINATE WORDS THAT APPEAR IN MORE THAN max_doc_freq OF DOCUMENTS (DOCUMENT ~ ROW)
#WILL GET RID OF COMMON WORDS SUCH AS "THE", "A", etc.
#LIMIT VOCABULARY TO max_feats COLUMNS (ONE FOR EACH WORD)
tfidfc = TfidfVectorizer(max_df=.50,max_features=60)
tf_revc = tfidfc.fit(feat_ptrc.copy())
vocab2c = tf_idfTokenizer(feat_ptrc,tf_revc)
#THOUGHT: TUNE THE HYPERPARAMETERS TO OPTIMIZE THE TOKENIZER?
vocab2c.head()


# In[1846]:


#DROP ORIGINAL STATE COLUMN AND LATER REPLACE WITH ENCODED MATRIX COLUMNS
feats_to_drop.append(column)
tokenize_cols.append(column)
input_jeeps.head()


# In[1847]:


vocabjs = pd.merge(vocab1j,vocab2j,left_index=True,right_index=True)
vocabcs = pd.merge(vocab1c,vocab2c,left_index=True,right_index=True)
vocabjs.head()


# In[1848]:


vocabcs.head()


# In[1849]:


col+=1
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj.value_counts())
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc.value_counts())


# In[1850]:


#ENTIRE COLUMN HAS VALUE "USED".....  DROPPING....
feats_to_drop.append(column)

col+=1 
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj.value_counts())
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc.value_counts())


# In[1851]:


#BASICALLY ALL 8-SPEED SO IT GETS DROPPED
feats_to_drop.append(column)

col+=1 
feat_ptrj,column = setFeatPtr(input_jeeps,col)
print(feat_ptrj.value_counts())
feat_ptrc,column = setFeatPtr(input_caddys,col)
print(feat_ptrc.value_counts())


# In[1852]:


#ONLY 5 UNIQUES IN OUR DATASET SO WE WILL ONE HOT ENCODE THE CATEGORIES
agej = calculate_age(feat_ptrj)
agec = calculate_age(feat_ptrc)

input_jeeps[column] =  agej
input_caddys[column] = agec

same_cols.append(column)

print(encoded_cols)
print(self_encodej)
print(self_encodec)
print(tokenize_cols)


# In[1853]:


feats_handled = (log_cols+encoded_cols+freq_cols+same_cols+mask_cols+tokenize_cols+orig_cols)
#print(orig_cols)
print("HANDLED FEATS:",feats_handled)
print("FEATS TO DROP:",feats_to_drop)


overlap = list(set(feats_handled) & set(feats_to_drop))

print("Overlapping elements:", overlap)
print("SIZE IS 26: ", len(feats_handled+feats_to_drop)-len(overlap)==26)


# In[1854]:


feats_to_drop = [col.strip().lower() for col in feats_to_drop]
input_jeeps.columns = [col.strip().lower() for col in input_jeeps.columns]
input_jeeps.drop(columns=feats_to_drop,inplace=True)
input_jeeps.head()


# In[1855]:


input_caddys.columns = [col.strip().lower() for col in input_caddys.columns]
input_caddys.drop(columns=feats_to_drop,inplace=True)
input_caddys.head()


# In[1856]:


input_jeeps = pd.merge(input_jeeps,self_encodej,left_index=True,right_index=True)
print(self_encodej.columns)
input_jeeps.info()


# In[1857]:


input_caddys = pd.merge(input_caddys,self_encodec,left_index=True,right_index=True)
print(self_encodec.columns)
input_caddys.info()


# In[1858]:


temp_encodedj = input_jeeps[encoded_cols]
print(encoded_cols)
print(temp_encodedj.columns)


# In[1859]:


encoderj = OneHotEncoder(handle_unknown='ignore')
coderj = encoderj.fit(temp_encodedj)
temp_encodedj.columns = temp_encodedj.columns.astype(str)
temp_encodedj = oHotEncode(temp_encodedj,coderj)
temp_encodedj.head()


# In[1860]:


temp_encodedc = input_caddys[encoded_cols]
print(temp_encodedc.columns)


# In[1861]:


encoderc = OneHotEncoder(handle_unknown='ignore')
coderc = encoderc.fit(temp_encodedc)
temp_encodedc.columns = temp_encodedc.columns.astype(str)
temp_encodedc = oHotEncode(temp_encodedc,coderc)
temp_encodedc.head()


# In[1862]:


input_jeeps.drop(columns=encoded_cols,inplace=True)
post_feat_engj = pd.merge(input_jeeps,temp_encodedj,left_index=True, right_index=True)
post_feat_engj.head()


# In[1863]:


input_caddys.drop(columns=encoded_cols,inplace=True)
post_feat_engc = pd.merge(input_caddys,temp_encodedc,left_index=True, right_index=True)
post_feat_engc.head()


# In[1864]:


post_feat_engj = pd.merge(post_feat_engj,vocabjs,left_index=True, right_index=True)
post_feat_engj.head()


# In[1865]:


post_feat_engc = pd.merge(post_feat_engc,vocabcs,left_index=True, right_index=True)
post_feat_engc.head()


# In[1866]:


types = post_feat_engj.select_dtypes(include=['object'])

# Display the object-type columns
print(types)


# In[1867]:


post_feat_engj["owners"] = pd.to_numeric(post_feat_engj["owners"], errors='coerce').fillna(0).astype(int)
print(post_feat_engj["owners"].value_counts())
post_feat_engj.columns = post_feat_engj.columns.astype(str)
post_feat_engj.info()


# In[1868]:


post_feat_engc["owners"] = pd.to_numeric(post_feat_engc["owners"], errors='coerce').fillna(0).astype(int)
print(post_feat_engc["owners"].value_counts())
post_feat_engc.columns = post_feat_engc.columns.astype(str)
post_feat_engc.info()


# In[1869]:


columnsj_missing = post_feat_engj.columns[post_feat_engj.isna().any()].tolist()

# Display columns with missing values
print("Columns with missing values:", columnsj_missing)


# In[1870]:


columnsc_missing = post_feat_engc.columns[post_feat_engc.isna().any()].tolist()

# Display columns with missing values
print("Columns with missing values:", columnsc_missing)


# In[1871]:


print(post_feat_engj.isna().sum().sum())
post_feat_engj.head()


# In[1872]:


print(post_feat_engc.isna().sum().sum())
post_feat_engc.head()


# In[1873]:


output_data = pd.DataFrame(df_train.iloc[:,-2:].copy())
output_jeeps = output_data[df_train["vehmake"] == "Jeep"].copy()
output_caddys = output_data[df_train["vehmake"] == "Cadillac"].copy()
print(output_jeeps["vehicle_trim"].value_counts())
print(output_caddys["vehicle_trim"].value_counts())


# In[1874]:


df_test.isna().sum()
test_df = df_test.copy()
test_jeeps = pd.DataFrame(test_df[test_df["VehMake"]=="Jeep"])
test_caddys = pd.DataFrame(test_df[test_df["VehMake"]=="Cadillac"])


# In[1875]:


#NOW APPLY THE SAME ENCODING AND TRANSFORMATIONS TO THE TEST DATASET
test_data_jeeps = engineerTestData(test_jeeps,log_cols,encoded_cols,freq_cols,
                             mask_cols,tokenize_cols,orig_cols,feats_to_drop,
                             coderj,tf_featsj,tf_revj)


# In[1876]:


#NOW APPLY THE SAME ENCODING AND TRANSFORMATIONS TO THE TEST DATASET
test_data_caddys = engineerTestData(test_caddys,log_cols,encoded_cols,freq_cols,
                             mask_cols,tokenize_cols,orig_cols,feats_to_drop,
                             coderc,tf_featsc,tf_revc)


# In[1877]:


print(test_data_jeeps.isna().sum().sum())
print(test_data_jeeps.info())
test_data_jeeps.columns


# In[1878]:


print(test_data_caddys.isna().sum().sum())
print(test_data_caddys.info())
test_data_caddys.head()


# In[1879]:


columns_with_missing_values = test_data_jeeps.columns[test_data_jeeps.isna().any()].tolist()
print(test_data_jeeps.index)
# Display columns with missing values
print("Columns with missing values:", columns_with_missing_values)


# In[1880]:


columns_with_missing_values = test_data_caddys.columns[test_data_caddys.isna().any()].tolist()
print(test_data_caddys.index)
# Display columns with missing values
print("Columns with missing values:", columns_with_missing_values)


# In[1881]:


test_data_jeeps["owners"] = pd.to_numeric(test_data_jeeps["owners"], errors='coerce').fillna(0).astype(int)
test_data_caddys["owners"] = pd.to_numeric(test_data_caddys["owners"], errors='coerce').fillna(0).astype(int)


# In[1882]:


print(test_data_jeeps.shape)
print(test_data_caddys.shape)
#test_data_jeeps = test_data_jeeps[post_feat_engj.columns]

post_feat_engj.to_csv("postfeatengj.csv")
test_data_jeeps.to_csv("testtransj.csv") 
columns_unique_to_df1 = set(post_feat_engj.columns) - set(test_data_jeeps.columns)
columns_unique_to_df2 = set(test_data_jeeps.columns) - set(post_feat_engj.columns)
common_columns = post_feat_engj.columns.intersection(test_data_jeeps.columns)

print("Columns unique to DataFrame 1:", columns_unique_to_df1)
print("Columns unique to DataFrame 2:", columns_unique_to_df2)
print("Common columns:", common_columns)


# In[1883]:


jeep_encoder = LabelEncoder()
caddy_encoder = LabelEncoder()

pre_encoded_jeeps = output_jeeps["vehicle_trim"]
pre_encoded_caddys = output_caddys["vehicle_trim"]

print(pre_encoded_jeeps.value_counts())
jeep_veh_trim = pd.Series(jeep_encoder.fit_transform(pre_encoded_jeeps),
                     index=post_feat_engj.index,
                     name=pre_encoded_jeeps.name)
caddy_veh_trim = pd.Series(caddy_encoder.fit_transform(pre_encoded_caddys),
                     index=pre_encoded_caddys.index,
                     name=pre_encoded_caddys.name)
print(np.unique(jeep_encoder.inverse_transform(jeep_veh_trim),return_counts=True))

list_pricej = output_jeeps["dealer_listing_price"]
list_pricec = output_caddys["dealer_listing_price"] 


# In[1884]:


post_feat_engj.drop("vehmake", axis=1, inplace=True)
print(jeep_veh_trim.value_counts())

post_feat_engc.drop("vehmake", axis=1, inplace=True)
print(caddy_veh_trim.value_counts())


# In[1885]:


clfj = tr.XGB_Classifier(post_feat_engj,jeep_veh_trim,False,jeep_encoder)
clfc = tr.XGB_Classifier(post_feat_engc,caddy_veh_trim,False,caddy_encoder)


# In[1886]:


regj = tr.XGB_Regressor(post_feat_engj,list_pricej,jeep_veh_trim,jeep_encoder,False,True)
regc = tr.XGB_Regressor(post_feat_engc,list_pricec,caddy_veh_trim,caddy_encoder,False,True)


# In[1887]:


#TEST VEHICLE TRIM PREDICTIONS
test_data_jeeps.drop("vehmake",axis=1,inplace=True)
clfj.prediction(test_data_jeeps)
print(clfj.preds.isna().sum())
print(clfj.preds.value_counts())

test_data_caddys.drop("vehmake",axis=1,inplace=True)
clfc.prediction(test_data_caddys)
print(clfc.preds.isna().sum())
print(clfc.preds.value_counts())


# In[1888]:


percentages = pd.DataFrame({"TEST PREDS" : clfj.preds.value_counts(normalize=True), 
                            "TRAINING LABELS" : pre_encoded_jeeps.value_counts(normalize=True),
                           "Delta": clfj.preds.value_counts(normalize=True)-pre_encoded_jeeps.value_counts(normalize=True)})
percentages


# In[1889]:


exp_pricesj = pr.calc_exp_prices(test_data_jeeps,pre_encoded_jeeps,regj)
exp_pricesc =  pr.calc_exp_prices(test_data_caddys,pre_encoded_caddys,regc)
print(exp_pricesj)


# In[1890]:


test_preds_pricej = pr.calc_test_prices(test_data_jeeps,clfj,regj,exp_pricesj)
test_preds_pricec = pr.calc_test_prices(test_data_caddys,clfc,regc,exp_pricesc)


# In[1891]:


#JEEPS WITHOUT POSTERIOR PROBABILITY
model_w_trims_preds = regj.prediction(pd.concat([test_data_jeeps,clfj.preds_proba],axis=1))
price_predsj_wtrims = pd.Series(model_w_trims_preds,index=test_data_jeeps.index,name=list_pricej.name)
#CADDYS WITHOUT POSTERIOR PROBABILITY
model_w_trims_predsc = regc.prediction(pd.concat([test_data_caddys,clfc.preds_proba],axis=1))
price_predsc_wtrims = pd.Series(model_w_trims_predsc,index=test_data_caddys.index,name=list_pricec.name)


# In[1892]:


#Train a new model without the trims involved
regj_no_trim = tr.XGB_Regressor(post_feat_engj,list_pricej,jeep_veh_trim,jeep_encoder,False,False)
regj_no_trim.prediction(test_data_jeeps)
no_trims = pd.Series(regj_no_trim.preds,index=test_data_jeeps.index,name=list_pricej.name)


# In[1893]:


desc_stats = pd.DataFrame({'Training Prices': list_pricej.describe(),
                           'Expected Price Prob Calculation': test_preds_pricej.describe(),
                          'Model Prediction with trim probs': price_predsj_wtrims.describe(),
                          'Trim Agnostic Preds': no_trims.describe()})

desc_stats


# In[1894]:


#Test Exp price calc on training data


# In[1895]:


'''import optuna

# Define the objective function for Optuna
def objective_trim(trial,X_train,y_train):
    # Sample hyperparameters from the search space
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    num_units = trial.suggest_int('num_units', 64, 256)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    hidden_activation = trial.suggest_categorical('hidden_activation', ['relu', 'tanh', 'sigmoid'])
    output_activation = trial.suggest_categorical('output_activation', ['softmax'])

    # Pack hyperparameters into a list
    params = [learning_rate, num_units, dropout_rate, num_layers, batch_size, hidden_activation, output_activation]

    # Call the objective function with the sampled hyperparameters
    return nm.objective_function(params, X_train, y_train, 'trim')

# Run Optuna optimization
study_trim = optuna.create_study(direction='minimize')
study_trim.optimize(lambda trial: objective_trim(trial, post_feat_engj, jeep_veh_trim), n_trials=50,n_jobs=-1)  # You can adjust the number of trials

# Get the best parameters from the optimization
best_params_trim = study_trim.best_params'''


# In[1896]:


'''best_score = study_trim.best_value

# Print the best score
print(f"Best Score: {best_score}")'''


# In[1897]:


# Concatenating predictions for Jeep
final_jeep_outputs = pd.concat([clfj.preds, test_preds_pricej], axis=1)

# Concatenating predictions for Caddy
final_caddy_outputs = pd.concat([clfc.preds, test_preds_pricec], axis=1)

# Concatenating final outputs and preserving the index order of test_df
final_outputs = pd.concat([final_jeep_outputs, final_caddy_outputs], axis=0)
final_outputs = final_outputs.sort_index()

# Naming columns appropriately
final_outputs.columns = output_data.columns

print(final_outputs)


# In[1898]:


final_outputs['Index'] = final_outputs.index
final_outputs = final_outputs[['Index',output_data.columns[0],output_data.columns[1]]]
final_outputs.head()


# In[1899]:


print(final_outputs.isna().sum())


# In[1900]:


final_outputs.to_csv('submission.csv', index=False, header=False)


# In[1901]:


xgb.plot_importance(clfj.model,max_num_features=25,importance_type='weight', show_values=False, xlabel='Importance', ylabel='Features', orientation='horizontal')
plt.show()


# In[1902]:


#Gathering stats using expected price calculation on training data splits
test_exp_j = tr.XGB_Regressor(post_feat_engj,list_pricej,jeep_veh_trim,jeep_encoder,True,True)

test_exp_c = tr.XGB_Regressor(post_feat_engc,list_pricec,caddy_veh_trim,caddy_encoder,True,True)


# In[1903]:


importance_scores = regc.model.get_booster().get_score(importance_type='gain')
sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:25]

# Separate feature names and scores for plotting
feature_names, scores = zip(*sorted_importance)

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_names)), scores, align='center')
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('Feature Importance Score (Gain)')
plt.title('Feature Importance Scores')
plt.show()


# In[ ]:




