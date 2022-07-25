import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Delete noisy feature (Precipitation) and rows that have >3 missing features
# We have the processed data after this function

def processing (df):
    df = df.drop(['P', 'CO2'], axis=1)
    df = df[df.isna().sum(axis=1) < 4]
    return df

# Reading data
df1 = pd.read_excel('./data/dataset.xlsx')

#Number of missing values
#df1.isna().sum()
#df1.isna().sum()[df1.isna().sum()>0].plot(kind='bar')

#Checking the correlations between features and target
def corr(df):
    pearson_list = []
    spearman_list = []
    kendall_list = []
    for col in df.iloc[:, 3:].columns:
        df2 = df[df[col].notna()]
        pearson = scipy.stats.pearsonr(df2[col], df2['NEE'])
        spearman = scipy.stats.spearmanr(df2[col], df2['NEE'])
        kendall = scipy.stats.kendalltau(df2[col], df2['NEE'])
        pearson_list.append(pearson)
        spearman_list.append(spearman)
        kendall_list.append(kendall)
    return pearson_list, spearman_list, kendall_list


# Saving the correlations with p values
#print(corr(df1))
      
#np.savetxt("correlation1.csv", corr(df1)[0], delimiter=",")
#np.savetxt("correlation2.csv", corr(df1)[1], delimiter=",")
#np.savetxt("correlation3.csv", corr(df1)[2], delimiter=",")

# Correlations between all variables
"""
corr = df1.corr(method = 'spearman')
fig, ax = plt.subplots(figsize=(15,10)) 
sns.heatmap(corr, annot = True, ax=ax)
plt.show()
"""


# Display a pairplot to look at relationships between variables
"""
plt.figure(figsize=(10,10))
sns.pairplot(data=df1,diag_kind='kde')
plt.show()
"""

def feature_engineering1(df):
    df['month'] = pd.DatetimeIndex(df['Date']).month
    df['season'] = df['month'] 
    df['season'] = df['season'].replace([2,12], 1)
    df['season'] = df['season'].replace([3,4,5], 2)
    df['season'] = df['season'].replace([6,7,8], 3)
    df['season'] = df['season'].replace([9,10,11], 4)
    return df

df1 = feature_engineering1(df1)

# Fill the continuous columns using median
missing_cols_median = ['SWC']
def missing_vals(df):
    for col in missing_cols_median:
        df.loc[:,col] = df.loc[:,col].fillna(df.loc[:,col].median())
    return df

df1 = missing_vals(df1)

def impute_missing(df):
    # Impute the missing data
    imputer = IterativeImputer(max_iter=10, random_state=0)
    df.iloc[:, 2:] = pd.DataFrame(imputer.fit_transform(df.iloc[:, 2:]),columns=df.iloc[:, 2:].columns)
    return df

df1 = impute_missing(df1)

def feature_engineering2(df):
    df = df.sort_values(by='Date', ascending=True)
    df['Date'] = df.groupby('Site')['Date'].apply(lambda x: x.sort_values())
    df['diff'] = df.groupby('Site')['Date'].diff() / np.timedelta64(1, 'D')
    df['WTD_diff'] = df['WTD'].diff()
    df.loc[df.loc[df['diff'] != 1].index.to_list(), 'WTD_diff'] = 0
    df = df.drop('diff', axis=1)
    return df

df1 = feature_engineering2(df1)

# Putting the target variable to the last column
df_col = df1.pop('NEE') 
df1['NEE']=df_col 

def onehot_encoder(df,cols):
    onehot_enc = OneHotEncoder(handle_unknown='ignore')
    onehot_enc.fit(df[cols])
    colnames = columns=list(onehot_enc.get_feature_names_out(input_features=cols))
    onehot_vals = onehot_enc.transform(df[cols]).toarray()
    enc_df = pd.DataFrame(onehot_vals,columns=colnames,index=df.index)
    df = pd.concat([df,enc_df],axis=1).drop(cols,axis=1)
    return df

onehotcols = ['Site', 'month']
df1 = onehot_encoder(df1,onehotcols)

#Defining the train and test sets
X = df1.drop(['Date','NEE'],axis=1)
y = df1['NEE']   
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=21,test_size=0.1)

train_set = pd.concat([X_train,y_train],axis=1)

# Checking anomalies for these columns:
df_columns = ['SW_IN', 'TA', 'WS', 'VPD', 'SWC', 'TS', 'WTD', 'WTD_diff']

def iqr(col,k):
    q25 = np.percentile(col,25)
    q75 = np.percentile(col,75)
    iqr = q75 - q25
    upper_limit = q75 + k * iqr
    lower_limit = q25 - k * iqr
    anomalies = list(col.index[(col>upper_limit) | (col<lower_limit)])
    return anomalies

def anomalies(df, df_columns):   
    all_anomalies_iqr = []
    k=3 #looking only for extreme outliers 
    for c in df_columns:
        anomalies_iqr = iqr(df[c],k)
        all_anomalies_iqr += anomalies_iqr
    # The list of all anomalous rows
    all_anomalies_iqr = set(all_anomalies_iqr)
    all_anomalies_iqr = sorted(all_anomalies_iqr)
    #print(len(all_anomalies_iqr))
    return all_anomalies_iqr

def remove_IQR_outliers(df):
    # Drop outliers
    df = df.drop(anomalies(train_set, df_columns),axis=0)
    return df

train_set = remove_IQR_outliers(train_set)

X_train = train_set.drop(['NEE'],axis=1)
y_train = train_set['NEE']  

def f_importance(X, y):    
    ftest = SelectKBest(score_func=f_regression, k='all')
    ftest.fit(X,y)
    f_scores = pd.DataFrame(ftest.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    f_scores = pd.concat([dfcolumns,f_scores],axis=1)
    f_scores.columns = ['Feature','F-Score']  
    f_scores = f_scores.sort_values(by='F-Score',ascending=False)
    return f_scores

# Plot scores
"""
plt.figure(figsize=(15,5))
plt.bar(x=f_importance(X_train,y_train)['Feature'],height=f_importance(X_train,y_train)['F-Score'])
plt.xticks(rotation=90)
plt.title('F-score of each feature')
plt.show()
"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)