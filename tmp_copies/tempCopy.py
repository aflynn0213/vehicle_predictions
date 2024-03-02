# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# -*- coding: utf-8 -*-

from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, ElasticNet

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (make_scorer,
                             classification_report,                             
                             mean_squared_error, 
                             r2_score, 
                             accuracy_score, 
                             roc_auc_score,
                             precision_score,
                             recall_score,
                             confusion_matrix)

from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm

from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, 
                                   PolynomialFeatures, StandardScaler,
                                   label_binarize)
from scipy.sparse import csr_matrix

from pdpbox import pdp

from xgboost import XGBRegressor, XGBClassifier

    
    

# +
def engineerTestData(df,log_cols,encoded_cols,freq_cols,
                     mask_cols,token_cols,orig_cols, feats_to_drop,
                     encoder,token_1,token_2):
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
    temp_encoded = oHotEncode(test_encode,encoder)
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
    
def oHotEncode(df,coder):
    encoded_mat = coder.transform(df)
    return pd.DataFrame(encoded_mat.todense(),
                        columns=[cat for columns in encoder.categories_ for cat in columns],
                        index=df.index)

    
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
    df = df.str.strip()
    # Applies one-hot encoding to the 'History' column based on the unique phrases
    encoded_df = df.str.get_dummies(',').reindex(columns=unique_phrases, fill_value=0)
    # Checks if all columns for the specified phrases contain zeros and create a 'None of the above' column
    encoded_df['None of the above'] = (encoded_df.sum(axis=1) == 0).astype(int)
    encoded_df.index = df.index
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


# +
#Initialize training and test dataframes

df_train = pd.read_csv('Training_DataSet.csv')
df_test = pd.read_csv('Test_Dataset.csv')

#Drop blank cells from training set to clean up data (contemplated using mean, median, or mode imputation,
#but will explore without corrupting the data and due to the large size of the dataset eliminating
#some rows should suffice
df_train.dropna(axis=0,how='any',inplace=True) #EXPLICIT CALL TO DROP ROWS WITH A SINGLE MISSING VALUE
                                               #(DEFAULT CALL DOES SAME)
# -

df_train.columns = df_train.columns.str.lower()
df_train.set_index('listingid',inplace=True)
df_train.info()

# +
#NOTICE THERES ONLY JEEPS AND CADILLACS IN DATA SET BRAKE THEM UP FURTHER TO SEE
#THE TRIMS SINCE TRIMS ARE USUALLY EXCLUSIVE TO MANUFACTURER LINE
jeeps = df_train[df_train['vehmake'].str.lower() == 'jeep'].copy()
caddy = df_train[df_train['vehmake'].str.lower() == 'cadillac'].copy()

print(jeeps['vehicle_trim'].value_counts())
print(caddy['vehicle_trim'].value_counts())

# +
#MASSIVE CLASS IMBALANCE WILL NEED TO CONDENSE THIS AND IGNORE LOW FREQUENCY CLASSES
#BECAUSE THEY ADD NOISE AND CLASSIFIER WILL NOT BE ABLE TO ARBITRATE
conditions = [
    caddy['vehicle_trim'].str.lower().str.contains('premium'),
    caddy['vehicle_trim'].str.lower().str.contains('luxury'),
    caddy['vehicle_trim'].str.lower().str.contains('base'),
    caddy['vehicle_trim'].str.lower().str.contains('platinum')
]

choices = ['Premium', 'Luxury', 'Base', 'Platinum']

# Use np.select() to relabel based on conditions
caddy['vehicle_trim'] = np.select(conditions, choices, default='other')

# Filter the DataFrame to keep only rows labeled as 'premium', 'luxury', 'base', or 'platinum'
valid_labels = ['Premium', 'Luxury', 'Base', 'Platinum']
caddy = caddy[caddy['vehicle_trim'].isin(valid_labels)]
caddy["vehicle_trim"]

# +
valid_labels_jeep = ['limited', 'laredo',  'summit',
                     'overland', 'altitude','trailhawk']

conditions_jeep = [
    jeeps['vehicle_trim'].str.lower().str.contains(label) for label in valid_labels_jeep
]

choices_jeep = ['Limited', 'Laredo',  'Summit', 
                     'Overland', 'Altitude','Trailhawk']

# Use np.select() to classify based on conditions
jeeps['vehicle_trim'] = np.select(conditions_jeep, choices_jeep, default='other')

# Filter the DataFrame to keep only rows labeled with valid labels
jeeps = jeeps[jeeps['vehicle_trim'].isin(choices_jeep)]
jeeps["vehicle_trim"]
# -

print("CADDY")
print(caddy["vehicle_trim"].value_counts())
print("JEEP")
print(jeeps["vehicle_trim"].value_counts())

print(jeeps.index)
print(caddy.index)
df_train.update(jeeps[['vehicle_trim']])
df_train.update(caddy[['vehicle_trim']])
df_train["vehicle_trim"].value_counts()

# +
options = choices + choices_jeep
df_train = df_train[df_train['vehicle_trim'].isin(options)]

print(df_train["vehicle_trim"].value_counts())
df_train.head()

# +
feats_to_drop = []
encoded_cols = []
freq_cols = []
same_cols = []
mask_cols = []
log_cols = [] 
orig_cols = []

input_data = df_train.copy()
input_data = input_data.iloc[:,:-2]

col = 0
feat_ptr,column = setFeatPtr(input_data,col)
# -

feat_ptr.head()

#PERCENTAGE MODE APPEARS
count = (feat_ptr==feat_ptr.mode()[0]).sum()
print(count/len(feat_ptr))
print(feat_ptr.nunique())

value_counts = feat_ptr.value_counts()
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
freq = feat_ptr.value_counts().to_dict()
feat_ptr = feat_ptr.map(freq)
feat_ptr.head()

input_data[column] = feat_ptr
input_data.head()

freq_cols.append(column)
col+=1
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()

feat_ptr.value_counts()

# +
feats_to_drop.append(column)

col+=1
feat_ptr,column = setFeatPtr(input_data,col)
print(feat_ptr.nunique())
feat_ptr.unique()

# +
#A CATEGORY COLUMN EASY TO ONE HOT ENCODE WITH A SMALL ENUMERATION AMOUNT (ONLY REQUIRES
# 5 COLUMNS TO ENCODE)
encoded_cols.append(column)
print(feat_ptr)

col+=1
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()
# -

print(feat_ptr.value_counts()[feat_ptr.mode()]/len(feat_ptr))
feat_ptr.value_counts().head(30)

#NOT CATEGORICAL OR CONTAINS DOMINATE VALUES, WILL NOT SIGNFICANTLY IMPACT MODEL PREDICTION EFFICIENCY
feats_to_drop.append(column)

col +=1
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()

#POSSIBLY NORMALIZE (Z-TRANSFORM) FOR NOW KEEP IT INTACT
same_cols.append(column)
col+=1
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()

plotDist(feat_ptr,'Density of Seller Review Count')
plotDist(np.log(feat_ptr),'Density of Log(Seller Review Count)')
plotDist(zScoreTransform(feat_ptr),'Density of Z-Transform of Seller Review Count')
plotDist(zScoreTransform(np.log(feat_ptr)),'Density of Z-Transform of Log of Seller Review Count')


# +
#KEEP REVIEW COUNT AS IS FOR NOW
same_cols.append(column)

col+=1 
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()

# +
#STATES -> CATEGORICAL
encoded_cols.append(column)

col+=1 
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.nunique()

# +
#ZIP SEEMS REDUNDANT WITH CITY/STATE INFO ALREADY EXISTING
#PLUS THE AMOUNT OF VARYING ZIPS PROVIDES NOISY DATA
feats_to_drop.append(column)
print(feats_to_drop)

col+=1
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()
# -

feat_ptr.nunique()

#ALL SUV, MEANINGLESS DATA
feats_to_drop.append(column)
col+=1
feat_ptr, column = setFeatPtr(input_data,col)
feat_ptr.head()

# +
#MASK BOOLEANS AS 1 AND 0's
feat_ptr = (feat_ptr).astype(int)
input_data[column] = feat_ptr
mask_cols.append(column)

col+=1
feat_ptr, column = setFeatPtr(input_data,col)
feat_ptr.head()
# -

feat_ptr.value_counts()

# +
temp_df = handle_vehcolorext(feat_ptr)
temp_df.columns = temp_df.columns.str.lower()

#Encoded with hand-written function rather than the encoder
self_encode = pd.DataFrame(temp_df, index=temp_df.index,columns=temp_df.columns)
orig_cols.append(column)
#Want to drop original
feats_to_drop.append(column)

print(temp_df.sum())
print(temp_df[temp_df["none"]==1].index)
# -

col+=1
feat_ptr, column = setFeatPtr(input_data,col)
feat_ptr.value_counts().head(50)

# +
temp_df = handle_vehcolorint(feat_ptr)
temp_df.columns = temp_df.columns.str.lower()

#Merge two handwritten encoded columns
self_encode = pd.merge(self_encode, temp_df, left_index=True, right_index=True)
orig_cols.append(column)
#Want to drop original
feats_to_drop.append(column)

print(temp_df.sum())
print(temp_df[temp_df["none"]==1].index)
# -

col+=1
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()

feat_ptr.value_counts()

#BASED OFF UNIQUE VALUES SEPERATE INTO 4WD,FWD,or AWD
temp_df = handle_vehdrivetrain(feat_ptr)
temp_df.value_counts()

encoded_cols.append(column)
input_data[column] = temp_df
orig_cols.append(column)
col+=1
print(encoded_cols)
print(input_data[column])

feat_ptr, column = setFeatPtr(input_data,col)
feat_ptr.value_counts()

#handle_vehengine takes the vehEngine column and turns it into a 
#2 column data frame by splitting the phrases into engine size 
#and cyclinder configuration
temp_df = handle_vehengine(feat_ptr)
print(temp_df["EngineSize"].value_counts())
print(temp_df["Cylinders"].value_counts())
# '0' represents unknown for either columns

input_data[temp_df.columns] = temp_df
orig_cols.append(column)
feats_to_drop.append(column)
col+=1
print(encoded_cols)
temp_df.head()

feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()

#ELIMINATE WORDS THAT APPEAR IN MORE THAN max_doc_freq OF DOCUMENTS (DOCUMENT ~ ROW)
#WILL GET RID OF COMMON WORDS SUCH AS "THE", "A", etc.
#LIMIT VOCABULARY TO max_feats COLUMNS (ONE FOR EACH WORD)
tf_feats = TfidfVectorizer(max_df=0.50,max_features=18)
temp_df = feat_ptr.copy()
tf_feats = tf_feats.fit(temp_df)
vocab1 = tf_idfTokenizer(temp_df,tf_feats)
#THOUGHT: TUNE THE HYPERPARAMETERS TO OPTIMIZE THE TOKENIZER?
vocab1.head()

#DROP ORIGINAL STATE COLUMN AND LATER REPLACE WITH ENCODED MATRIX COLUMNS
feats_to_drop.append(column)
tokenize_cols = [column]
input_data.head()

col+=1
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.value_counts()

# +
encoded_cols.append(column)

col+=1

# +
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()

temp_df = feat_ptr.str.split(',',n=1,expand=True)
temp_df.columns = ['Owners', 'History']
temp_df["History"].unique()
# -

temp_df['Owners'] = temp_df['Owners'].str.extract(r'^(\d+)')
temp_df['Owners'].head()

input_data['Owners'] = temp_df['Owners']
input_data['Owners'].head()

temp_df["History"].value_counts()

#TURNS OUT THAT THESE PHRASES CAN ACTUALLY BE TURNED INTO CATEGORICAL COLUMNS
#EACH ELEMENT IS A COMBINATION OF VARYING SIZE OF THE 4 POSSIBLE UNIQUE PHRASES
#ONE HOT ENCODE WITH A COLUMN FOR EACH PHRASE
encoded_hist = handle_vehhistory(temp_df["History"])
encoded_hist.head()

#DROP ORIGINAL COLUMN AND LATER REPLACE WITH ENCODED MATRIX COLUMNS
feats_to_drop.append(column)
self_encode = pd.merge(self_encode, encoded_hist, left_index=True, right_index=True)
orig_cols.append(column)
self_encode.head()

col+=1
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()

#Use ceiling in order to round to whole days and start the listings 
#on day 1 rather than day 0
feat_ptr = pd.Series(np.ceil(feat_ptr),index=feat_ptr.index)
feat_ptr.head()

plotDist(feat_ptr,"Distribution of Listing Days Frequency")
plotDist(np.log(feat_ptr),"Distribution of Log(Listing Days) Frequency")
plotDist(zScoreTransform(feat_ptr),"Distribution of Z-Transform(Listing Days) Frequency")
plotDist(zScoreTransform(np.log(feat_ptr)),"Distribution of Z-Tranform(Log(Listing Days)) Frequency")


#CHOOSE LOG VALUE
feat_ptr = np.log(feat_ptr)
feat_ptr.head()

# +
input_data[column] = feat_ptr
log_cols.append(column)

col+=1
input_data.head()
# -

feat_ptr, column = setFeatPtr(input_data,col)
feat_ptr.head()

# +
encoded_cols.append(column)

col+=1
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()
# -

plotDist(feat_ptr,"Density of Vehicle Mileage")
plotDist(np.log(feat_ptr),"Density of Log(Vehicle Mileage)")
plotDist(zScoreTransform(feat_ptr),"Density of Z-Transform(Vehicle Mileage)")
plotDist(zScoreTransform(np.log(feat_ptr)),"Density of Z-Tranform(Log(Vehicle Mileage))")

# +
#ORIGINAL DATA LOOKS ~NORMAL~

same_cols.append(column)
col+=1
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()

# +
#ALREADY HAVE JEEP/CADILLAC ENCODED COLUMNS WHICH HAVE A DIRECT CORRELATION TO THIS
#WILL REMOVE THIS EXTRANEOUS COLUMN
feats_to_drop.append(column)

col+=1 
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.value_counts()

# +
encoded_cols.append(column)

col+=1
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.head()
# -

#ELIMINATE WORDS THAT APPEAR IN MORE THAN max_doc_freq OF DOCUMENTS (DOCUMENT ~ ROW)
#WILL GET RID OF COMMON WORDS SUCH AS "THE", "A", etc.
#LIMIT VOCABULARY TO max_feats COLUMNS (ONE FOR EACH WORD)
tfidf = TfidfVectorizer(max_df=.50,max_features=50)
tf_rev = tfidf.fit(feat_ptr.copy())
vocab2 = tf_idfTokenizer(feat_ptr,tf_rev)
#THOUGHT: TUNE THE HYPERPARAMETERS TO OPTIMIZE THE TOKENIZER?
vocab2.head()

#DROP ORIGINAL STATE COLUMN AND LATER REPLACE WITH ENCODED MATRIX COLUMNS
feats_to_drop.append(column)
tokenize_cols.append(column)
vocabs = pd.merge(vocab1,vocab2,left_index=True,right_index=True)
input_data.head()

col+=1
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.value_counts()

# +
#ENTIRE COLUMN HAS VALUE "USED".....  DROPPING....
feats_to_drop.append(column)

col+=1 
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.value_counts()

# +
#BASICALLY ALL 8-SPEED SO IT GETS DROPPED
feats_to_drop.append(column)

col+=1 
feat_ptr,column = setFeatPtr(input_data,col)
feat_ptr.value_counts()

# +
#ONLY 5 UNIQUES IN OUR DATASET SO WE WILL ONE HOT ENCODE THE CATEGORIES
encoded_cols.append(column)

print(encoded_cols)
print(self_encode)
print(tokenize_cols)

# +
feats_handled = (log_cols+encoded_cols+freq_cols+same_cols+mask_cols+tokenize_cols+orig_cols)

print(feats_handled)
print(feats_to_drop)

print(len(feats_handled+feats_to_drop))
# -

feats_to_drop = [col.strip().lower() for col in feats_to_drop]
input_data.columns = [col.strip().lower() for col in input_data.columns]
input_data.drop(columns=feats_to_drop,inplace=True)
input_data.head()

input_data = pd.merge(input_data,self_encode,left_index=True,right_index=True)
print(self_encode.columns)
input_data.head()

temp_encoded = input_data[encoded_cols]
print(encoded_cols)
print(temp_encoded.columns)
input_data.head(20)

encoder = OneHotEncoder(handle_unknown='ignore')
coder = encoder.fit(temp_encoded)
temp_encoded.columns = temp_encoded.columns.astype(str)
temp_encoded = oHotEncode(temp_encoded,coder)
temp_encoded.head()

input_data.drop(columns=encoded_cols,inplace=True)
post_feat_eng = pd.merge(input_data,temp_encoded,left_index=True, right_index=True)
post_feat_eng.head()

post_feat_eng = pd.merge(post_feat_eng,vocabs,left_index=True, right_index=True)
post_feat_eng.head()

# +
types = post_feat_eng.select_dtypes(include=['object'])

# Display the object-type columns
print(types)
# -

post_feat_eng["owners"] = pd.to_numeric(post_feat_eng["owners"], errors='coerce').fillna(0).astype(int)
print(post_feat_eng["owners"].value_counts())
post_feat_eng.columns = post_feat_eng.columns.astype(str)
post_feat_eng.info()

# +
columns_with_missing_values = post_feat_eng.columns[post_feat_eng.isna().any()].tolist()

# Display columns with missing values
print("Columns with missing values:", columns_with_missing_values)
# -

print(post_feat_eng.isna().sum().sum())
post_feat_eng.head()

output_data = pd.DataFrame(df_train.iloc[:,-2:].copy())
output_data.head()

label_encoder = LabelEncoder()
pre_encoded_trim = output_data.iloc[:,0]
print(pre_encoded_trim.isna().sum())
pre_encoded_trim.value_counts()

veh_trim = pd.Series(label_encoder.fit_transform(pre_encoded_trim),
                     index=output_data.index,
                     name=pre_encoded_trim.name)
sellers_price = output_data.iloc[:,1]
print(veh_trim.isna().sum())
print(veh_trim.value_counts())
print(veh_trim.info())
veh_trim.head()

print(sellers_price.isna().sum())
print(sellers_price.value_counts())
print(sellers_price.info())
sellers_price.head()

df_test.isna().sum()

#NOW APPLY THE SAME ENCODING AND TRANSFORMATIONS TO THE TEST DATASET
test_data = engineerTestData(df_test,log_cols,encoded_cols,freq_cols,
                             mask_cols,tokenize_cols,orig_cols,feats_to_drop,
                             coder,tf_feats,tf_rev)

print(test_data.isna().sum().sum())
print(test_data.info())
test_data.head()

columns_with_missing_values = test_data.columns[test_data.isna().any()].tolist()
print(test_data.index)
# Display columns with missing values
print("Columns with missing values:", columns_with_missing_values)

test_data["owners"] = pd.to_numeric(test_data["owners"], errors='coerce').fillna(0).astype(int)


# +
print(test_data.shape)
print(post_feat_eng.shape)
test_data = test_data[post_feat_eng.columns]
test_data.shape
post_feat_eng.to_csv("postfeateng.csv")
test_data.to_csv("testtrans.csv") 
columns_unique_to_df1 = set(post_feat_eng.columns) - set(test_data.columns)
columns_unique_to_df2 = set(test_data.columns) - set(post_feat_eng.columns)
common_columns = post_feat_eng.columns.intersection(test_data.columns)

print("Columns unique to DataFrame 1:", columns_unique_to_df1)
print("Columns unique to DataFrame 2:", columns_unique_to_df2)
print("Common columns:", common_columns)
# -

'''param_grid = {
    'n_estimators': [100, 200],  # Number of trees in the forest
    'max_depth': [10,15],  # Maximum depth of each tree
    'min_samples_split': [5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [2, 4]  # Minimum number of samples required at each leaf node
}

param_grid_class = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 15],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': [None, 'log2'],
    'criterion': ['entropy']
}

print("REGRESSION")
rf = RandomForestRegressor()

gs_cv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

gs_cv.fit(post_feat_eng,sellers_price)
best_est = gs_cv.best_estimator_
print(" Best Score: ", gs_cv.best_score_)
print(" Best Parameters: ",gs_cv.best_params_)
feature_importances = best_est.feature_importances_
feature_importance_pairs = list(zip(input_data.columns, feature_importances))

# Sort feature importances in descending order
sorted_feature_importance = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

#Displays the Importance by calculating the Features that attributed the highest increase in uncertainty
#when removed.  Sum of importances over all features is 1
for feature, importance in sorted_feature_importance:
    print(f"Feature : {feature}, Regression Importance: {importance}")

print("CLASSIFICATION")
rf_cl = RandomForestClassifier()
gs_cv_cl = GridSearchCV(estimator=rf_cl, param_grid=param_grid_class, cv=5, scoring='roc_auc', n_jobs=-1)

gs_cv_cl.fit(input_data,veh_trim)
best_est = gs_cv_cl.best_estimator_
print(" Class Best Score: ", gs_cv_cl.best_score_)
print(" CLASS Best Parameters: ",gs_cv_cl.best_params_)
feature_importances = best_est.feature_importances_
feature_importance_pairs = list(zip(input_data.columns, feature_importances))

# Sort feature importances in descending order
sorted_feature_importance = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

#Displays the Importance by calculating the Features that attributed the highest increase in uncertainty
#when removed.  Sum of importances over all features is 1
for feature, importance in sorted_feature_importance:
    print(f"Feature : {feature}, Classification Importance: {importance}")
'''

# +
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}
xgb = XGBRegressor()
# Grid search using cross-validation
cv = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=4, scoring='r2')
cv.fit(post_feat_eng,sellers_price)

# Best parameters and best score
print("Best Parameters: ", cv.best_params_)
print("Best Score R2: ", cv.best_score_)

best_xgb_reg = cv.best_estimator_


# +
pars_clf = {
    'n_estimators': [200],
    'max_depth': [5, 7],
    'learning_rate': [0.05, 0.1]
}

xgb_clf = XGBClassifier()

scoring = {
    'precision': make_scorer(precision_score, average='weighted'),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr'),
    'accuracy': make_scorer(accuracy_score),
    'recall': make_scorer(recall_score,average='weighted')
}
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

xgb_cv = GridSearchCV(xgb_clf,param_grid=pars_clf,scoring=scoring,cv=skf, refit='roc_auc')
xgb_cv.fit(post_feat_eng,veh_trim)

# Best parameters and best score
print("Best Parameters: ", xgb_cv.best_params_)
print("Best Score: ", xgb_cv.best_score_)

# Evaluating on test data using best estimator
best_xgb_clf = xgb_cv.best_estimator_
# -

for metric in scoring:
    score_key = f"mean_test_{metric}"  # Adjust the key to access cv_results_
    score = xgb_cv.cv_results_[score_key]
    print(f"METRIC '{metric}': SCORES '{score}'")


#TEST VEHICLE TRIM PREDICTIONS
trim_preds = best_xgb_clf.predict(test_data)
trim_label_pred = pd.Series(label_encoder.inverse_transform(trim_preds),
                           index=test_data.index,
                           name=veh_trim.name)
print(trim_label_pred.isna().sum())
print(trim_label_pred.value_counts())

#TEST DEALERS LISTING PRICE
price_pred = best_xgb_reg.predict(test_data)
price_pred = pd.Series(price_pred,index=test_data.index,name=sellers_price.name)
print(price_pred.isna().sum())
price_pred.head()

results_df = pd.merge(trim_label_pred,price_pred,left_index=True,right_index=True)
results_df.to_csv('results_df.csv')

results_df['Index'] = results_df.index
results_df = results_df[['Index',trim_label_pred.name,price_pred.name]]
results_df.head()

print(results_df.isna().sum())

results_df.to_csv('submission.csv', index=False, header=False)















# +
# Define KFold with 5 folds
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_number = 1
for train_index, test_index in kfold.split(post_feat_eng):
    # Split data into train and test sets for this fold
    X_test = post_feat_eng.iloc[test_index]
    y_test = veh_trim.iloc[test_index]
    
    # Use the best estimator for prediction on the test data for this fold
    y_pred = best_xgb_clf.predict(X_test)
    
    # Calculate precision, recall, and ROC AUC scores
    precision = precision_score(y_test, y_pred,average='weighted')
    recall = recall_score(y_test, y_pred,average='weighted')
    roc_auc = roc_auc_score(y_test, best_xgb_clf.predict_proba(X_test),multi_class='ovr')
    
    # Generate confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Fold {fold_number} - Confusion Matrix:")
    print(cm)
    print(f"\nFold {fold_number} - Classification Report:")
    print(report)
    print(f"Fold {fold_number} - Precision: {precision}")
    print(f"Fold {fold_number} - Recall: {recall}")
    print(f"Fold {fold_number} - ROC AUC: {roc_auc}")
    print("-------------------------------------")
    
    fold_number += 1
# -


