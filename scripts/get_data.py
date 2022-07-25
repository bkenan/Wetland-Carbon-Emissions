import numpy as np
import pandas as pd


# Reading all the necessary files
xls = pd.ExcelFile('summary_wetland_gapfilled.xlsx')
df1 = pd.read_excel(xls, 'DPW')
df2 = pd.read_excel(xls, 'Elm')
df3 = pd.read_excel(xls, 'Esm')
df4 = pd.read_excel(xls, 'HB1')
df5 = pd.read_excel(xls, 'KS3')
df6 = pd.read_excel(xls, 'KS4')
df7 = pd.read_excel(xls, 'LA1')
df8 = pd.read_excel(xls, 'LA2')
df9 = pd.read_excel(xls, 'NC1')
df10 = pd.read_excel(xls, 'NC2')
df11 = pd.read_excel(xls, 'NC4')
df12 = pd.read_excel(xls, 'Skr')
df13 = pd.read_excel(xls, 'xDL')
df2_wtd = pd.read_csv('Elm.csv') 
df3_wtd = pd.read_csv('Esm.csv') 
df4_wtd = pd.read_csv('HB1.csv') 
df7_wtd = pd.read_csv('LA1.csv') 
df8_wtd = pd.read_csv('LA2.csv') 
df9_wtd = pd.read_csv('NC1.csv') 
df10_wtd = pd.read_csv('NC2.csv') 
df11_wtd = pd.read_csv('NC4.csv') 
df12_wtd = pd.read_csv('Skr.csv') 



def date_cleaning(df):
    df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
    df['TIMESTAMP_END'] = pd.to_datetime(df['TIMESTAMP_END'], format='%Y%m%d%H%M')
    df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START']).dt.date
    df['TIMESTAMP_END'] = pd.to_datetime(df['TIMESTAMP_END']).dt.date
    df.index = df['TIMESTAMP_START']
    df.index.names = ['Date']
    df = df.replace(-9999, np.NaN)
    df = df.replace(-10000, np.NaN)
    return df



def wtd_date(df):
    df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d')
    df['Time'] = pd.to_datetime(df['Time']).dt.date
    df.index = df['Time']
    df.index.names = ['Date']
    return df


def label(df):
    remove_NEE = np.unique(df[df['NEE'].isnull()].index.tolist())
    df = df.drop(np.unique(remove_NEE),axis=0)
    return df


def features(df):
    for col in df.iloc[:, 3:].columns:
        nan_index = df[df[col].isnull()].index.tolist()
        remove_nan = list(set([x for x in nan_index if nan_index.count(x)>8]))
        df.loc[remove_nan, col] = np.NaN
    return df


df1 = date_cleaning(df1)
df1 = label(df1)
df1 = features(df1)
df1 = df1[df1.index<pd.to_datetime('2016-01-06')]
df1 = df1.groupby(pd.to_datetime(df1.index).date).mean()
df1.insert(0, 'Site', 'DPW')
df1.isna().sum()


df2 = date_cleaning(df2)
df2 = label(df2)
df2 = features(df2)
df2_wtd = wtd_date(df2_wtd)
df2 = df2.groupby(pd.to_datetime(df2.index).date).mean()
df2['CO2'] = df2[['CO2_1', 'CO2_2']].mean(axis=1)
df2['TS'] = df2[['TS_1', 'TS_2']].mean(axis=1)
df2 = df2.join(df2_wtd)
df2 = df2.drop(columns=['TS_1', 'TS_2', 'CO2_1', 'CO2_2', 'Time'])
df2.insert(0, 'Site', 'Elm')


df3 = df3.rename({'NEE_PI': 'NEE'}, axis='columns')
df3 = date_cleaning(df3)
df3 = label(df3)
df3 = features(df3)
df3_wtd = wtd_date(df3_wtd)
df3 = df3.groupby(pd.to_datetime(df3.index).date).mean()
df3['CO2'] = df3[['CO2_1', 'CO2_2']].mean(axis=1)
df3['TS'] = df3[['TS_1', 'TS_2']].mean(axis=1)
df3 = df3.join(df3_wtd)
df3 = df3.drop(columns=['TS_1', 'TS_2', 'CO2_1', 'CO2_2', 'Time'])
df3.insert(0, 'Site', 'Esm')


df4 = date_cleaning(df4)
df4 = label(df4)
df4 = features(df4)
df4_wtd = wtd_date(df4_wtd)
df4 = df4.groupby(pd.to_datetime(df4.index).date).mean()
df4['TS'] = df4[['TS_1_1_1', 'TS_1_2_1']].mean(axis=1)
df4 = df4.join(df4_wtd)
df4 = df4.drop(columns=['TS_1_1_1', 'TS_1_2_1', 'Time'])
df4.insert(0, 'Site', 'HB1')


df5 = date_cleaning(df5)
df5 = label(df5)
df5 = features(df5)
df5 = df5.groupby(pd.to_datetime(df5.index).date).mean()
df5.insert(0, 'Site', 'KS3')


df6 = date_cleaning(df6)
df6 = label(df6)
df6 = features(df6)
df6 = df6[df6.index<pd.to_datetime('2019-12-29')]
df6 = df6.groupby(pd.to_datetime(df6.index).date).mean()
df6.insert(0, 'Site', 'KS4')


df7 = date_cleaning(df7)
df7 = label(df7)
df7 = features(df7)
df7_wtd = wtd_date(df7_wtd)
df7 = df7.groupby(pd.to_datetime(df7.index).date).mean()
df7 = df7.join(df7_wtd)
df7 = df7.drop(columns=['Time'])
df7.insert(0, 'Site', 'LA1')


df8 = date_cleaning(df8)
df8 = label(df8)
df8 = features(df8)
df8_wtd = wtd_date(df8_wtd)
df8 = df8[df8.index<pd.to_datetime('2013-11-18')]
df8 = df8.groupby(pd.to_datetime(df8.index).date).mean()
df8 = df8.join(df8_wtd)
df8 = df8.drop(columns=['Time'])
df8.insert(0, 'Site', 'LA2')


df9 = date_cleaning(df9)
df9 = label(df9)
df9 = features(df9)
df9_wtd = wtd_date(df9_wtd)
df9 = df9.groupby(pd.to_datetime(df9.index).date).mean()
df9 = df9.rename({'SW_IN_1_1_1': 'SW_IN'}, axis='columns')
df9 = df9.rename({'WS_1_1_1': 'WS'}, axis='columns')
df9 = df9.rename({'SWC_1_1_1': 'SWC_1'}, axis='columns')
df9 = df9.rename({'CO2_1_1_1': 'CO2'}, axis='columns')
df9['TS'] = df9[['TS_1_1_1', 'TS_1_2_1']].mean(axis=1)
df9 = df9.join(df9_wtd)
df9 = df9.drop(columns=['TS_1_1_1', 'TS_1_2_1', 'Time'])
df9.insert(0, 'Site', 'NC1')
df9



def features10_1(df):
    for col in df.iloc[:, 3:11].columns:
        nan_index = df[df[col].isnull()].index.tolist()
        remove_nan = list(set([x for x in nan_index if nan_index.count(x)>8]))
        df.loc[remove_nan, col] = np.NaN
    return df


def features10_2(df):
    index_list = df.index.tolist()
    for col in df.iloc[:, 11:].columns:
        nonnan_index = df[df[col].notnull()].index.tolist()
        valid_index  = list(set([x for x in nonnan_index if nonnan_index.count(x)>=40]))
        remove_nan = list(set(index_list) - set(valid_index))
        df.loc[remove_nan,col] = np.NaN
    return df


df10 = date_cleaning(df10)
df10 = label(df10)
df10 = features10_1(df10)
df10 = features10_2(df10)
df10_wtd = wtd_date(df10_wtd)
df10 = df10.groupby(pd.to_datetime(df10.index).date).mean()
df10 = df10.rename({'SW_IN_1_1_1': 'SW_IN'}, axis='columns')
df10 = df10.rename({'TA_1_1_1': 'TA'}, axis='columns')
df10 = df10.rename({'WS_1_1_1': 'WS'}, axis='columns')
df10 = df10.rename({'SWC_1_1_1': 'SWC_1'}, axis='columns')
df10['TS'] = df10[['TS_1_1_1', 'TS_1_2_1']].mean(axis=1)
df10['CO2'] = df10[['CO2_2_2_1', 'CO2_2_3_1', 'CO2_2_4_1', 'CO2_2_5_1']].mean(axis=1)
df10 = df10.join(df10_wtd)
df10 = df10.drop(columns=['TS_1_1_1', 'TS_1_2_1', 'WTD_1_1_1', 'CO2_2_2_1', 'CO2_2_3_1', 'CO2_2_4_1', 'CO2_2_5_1', 'Time'])
df10.insert(0, 'Site', 'NC2')
df10





df_col = df11.pop('SW_IN_1_1_1') 
df11['SW_IN_1_1_1']=df_col 

def features11_1(df):
    for col in df.iloc[:, 3:10].columns:
        nan_index = df[df[col].isnull()].index.tolist()
        remove_nan = list(set([x for x in nan_index if nan_index.count(x)>8]))
        df.loc[remove_nan, col] = np.NaN
    return df


def features11_2(df):
    index_list = df.index.tolist()
    for col in df.iloc[:, 10:].columns:
        nonnan_index = df[df[col].notnull()].index.tolist()
        valid_index  = list(set([x for x in nonnan_index if nonnan_index.count(x)>=40]))
        remove_nan = list(set(index_list) - set(valid_index))
        df.loc[remove_nan,col] = np.NaN
    return df


df11 = date_cleaning(df11)
df11 = label(df11)
df11 = features11_1(df11)
df11 = features11_2(df11)
df11_wtd = wtd_date(df11_wtd)
df11 = df11.groupby(pd.to_datetime(df11.index).date).mean()
df11 = df11.rename({'SW_IN_1_1_1': 'SW_IN'}, axis='columns')
df11 = df11.rename({'WS_1_1_1': 'WS'}, axis='columns')
df11 = df11.rename({'SWC_1_1_1': 'SWC_1'}, axis='columns')
df11['TS'] = df11[['TS_1_1_1', 'TS_1_2_1']].mean(axis=1)
df11['CO2'] = df11[['CO2_2_2_1', 'CO2_2_3_1', 'CO2_2_4_1', 'CO2_2_5_1']].mean(axis=1)
df11 = df11.join(df11_wtd)
df11 = df11.drop(columns=['TS_1_1_1', 'TS_1_2_1', 'WTD_1_1_1', 'CO2_2_2_1', 'CO2_2_3_1', 'CO2_2_4_1', 'CO2_2_5_1', 'Time'])
df11.insert(0, 'Site', 'NC4')
df_col2 = df11.pop('SW_IN') 
df11.insert(2, 'SW_IN', df_col2)
df11




df12 = date_cleaning(df12)
df12 = label(df12)
df12 = features(df12)
df12_wtd = wtd_date(df12_wtd)
df12 = df12.groupby(pd.to_datetime(df12.index).date).mean()
df12['CO2'] = df12[['CO2_1', 'CO2_2']].mean(axis=1)
df12['TS'] = df12[['TS_1', 'TS_2']].mean(axis=1)
df12 = df12.join(df12_wtd)
df12 = df12.drop(columns=['CO2_1', 'CO2_2', 'TS_1', 'TS_2', 'Time'])
df12.insert(0, 'Site', 'Skr')


df13 = date_cleaning(df13)
df13 = label(df13)
df13 = features(df13)
df13 = df13[df13.index<pd.to_datetime('2020-10-31')]
df13 = df13.groupby(pd.to_datetime(df13.index).date).mean()
df13['WS'] = df13[['WS_1_1_1',
 'WS_1_2_1',
 'WS_1_3_1',
 'WS_1_4_1',
 'WS_1_5_1',
 'WS_1_6_1']].mean(axis=1)
df13['TS'] = df13[['TS_1_1_1',
 'TS_1_2_1',
 'TS_1_3_1',
 'TS_1_4_1',
 'TS_1_5_1',
 'TS_1_6_1',
 'TS_1_7_1',
 'TS_1_8_1',
 'TS_1_9_1',
 'TS_2_1_1',
 'TS_2_2_1',
 'TS_2_3_1',
 'TS_2_4_1',
 'TS_2_5_1',
 'TS_2_6_1',
 'TS_2_7_1',
 'TS_2_8_1',
 'TS_2_9_1',
 'TS_3_1_1',
 'TS_3_2_1',
 'TS_3_3_1',
 'TS_3_4_1',
 'TS_3_5_1',
 'TS_3_6_1',
 'TS_3_7_1',
 'TS_3_8_1',
 'TS_3_9_1',
 'TS_4_1_1',
 'TS_4_2_1',
 'TS_4_3_1',
 'TS_4_4_1',
 'TS_4_5_1',
 'TS_4_6_1',
 'TS_4_7_1',
 'TS_4_8_1',
 'TS_4_9_1',
 'TS_5_1_1',
 'TS_5_2_1',
 'TS_5_3_1',
 'TS_5_4_1',
 'TS_5_5_1',
 'TS_5_6_1',
 'TS_5_7_1',
 'TS_5_8_1',
 'TS_5_9_1']].mean(axis=1)
df13['SWC_1'] = df13[['SWC_1_1_1',
 'SWC_1_2_1',
 'SWC_1_3_1',
 'SWC_1_4_1',
 'SWC_1_5_1',
 'SWC_1_6_1',
 'SWC_1_7_1',
 'SWC_1_8_1',
 'SWC_2_1_1',
 'SWC_2_2_1',
 'SWC_2_3_1',
 'SWC_2_4_1',
 'SWC_2_5_1',
 'SWC_2_6_1',
 'SWC_2_7_1',
 'SWC_2_8_1',
 'SWC_3_1_1',
 'SWC_3_2_1',
 'SWC_3_3_1',
 'SWC_3_4_1',
 'SWC_3_5_1',
 'SWC_3_6_1',
 'SWC_3_7_1',
 'SWC_3_8_1',
 'SWC_4_1_1',
 'SWC_4_2_1',
 'SWC_4_3_1',
 'SWC_4_4_1',
 'SWC_4_5_1',
 'SWC_4_6_1',
 'SWC_4_7_1',
 'SWC_4_8_1',
 'SWC_5_1_1',
 'SWC_5_2_1',
 'SWC_5_3_1',
 'SWC_5_4_1',
 'SWC_5_5_1',
 'SWC_5_6_1',
 'SWC_5_7_1',
 'SWC_5_8_1']].mean(axis=1)
df13['CO2'] = df13[['CO2_1_1_1',
 'CO2_1_1_2',
 'CO2_1_1_3',
 'CO2_1_2_2',
 'CO2_1_2_3',
 'CO2_1_3_2',
 'CO2_1_3_3',
 'CO2_1_4_2',
 'CO2_1_4_3',
 'CO2_1_5_2',
 'CO2_1_5_3',
 'CO2_1_6_2',
 'CO2_1_6_3']].mean(axis=1)
df13 = df13[['NEE','SW_IN', 'TA', 'VPD', 'WS', 'TS', 'SWC_1', 'CO2']]
df13.insert(0, 'Site', 'xDL')


frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13]
df = pd.concat(frames)
df = df.sort_index()
df.isna().sum()
df.shape

df.to_excel("final_dataset.xlsx")


