import numpy as np
import pandas as pd


df1 = pd.read_csv('./OCO2/Time.txt', sep=" ", header=None)
df1.columns = ["Time"]
df2 = pd.read_csv('./OCO2/Lat.txt', sep=" ", header=None)
df2.columns = ["Lat"]
df3 = pd.read_csv('./OCO2/Lon.txt', sep=" ", header=None)
df3.columns = ["Lon"]
df4 = pd.read_csv('./OCO2/XCO2.txt', sep=" ", header=None)
df4.columns = ["CO2"]

#Merge together
df_oco2 = pd.concat([df1, df2, df3, df4], axis=1)
df_oco2

#Cleaning

df_date = df_oco2['Time'].str.split(',', expand=True)
df_date = df_date.loc[:,0:2]
df_date.columns =['Year', 'Month', 'Day']
df_date['Date'] = pd.to_datetime(df_date[['Year', 'Month', 'Day']])
df_date = df_date['Date']
df_oco2 = df_oco2[["Lat", 'Lon', 'CO2']]
df_oco2 = pd.concat([df_date, df_oco2], axis=1)


#Adding data
xls1 = pd.ExcelFile('OCO3.xlsx')
df_oco3 = pd.read_excel(xls1, 'OCO3')
df_oco3['Date'] = pd.to_datetime(df_oco3[['Year', 'Month', 'Day']])
df_oco3 = df_oco3.rename(columns={'CO2/ppm': 'CO2'})
df_oco3 = df_oco3[['Date', "Lat", 'Lon', 'CO2']]

#Merging
df_oco = pd.concat([df_oco2, df_oco3])
df_oco

#Adding data
xls2 = pd.ExcelFile('GOSAT.xlsx')
df_gosat = pd.read_excel(xls2, 'GOSAT')
df_gosat['Date'] = pd.to_datetime(df_gosat[['Year', 'Month', 'Day']])
df_gosat = df_gosat.rename(columns={'CO2/ppm': 'CO2'})
df_gosat = df_gosat[['Date', "Lat", 'Lon', 'CO2']]
df_gosat


df = pd.read_excel('final_dataset48_40.xlsx')
flux_site_loc = pd.read_excel('flux_site_loc.xls')
flux_site_loc


def func(col, df):
    #df_co2_null = df[df['CO2'].isnull()]
    #df_co2_null2 = df_co2_null[df_co2_null['Site'] == col]
    df_gosat1 = df.loc[(df['Lat'] >= (flux_site_loc.loc[flux_site_loc['Site'] == col]['Y'].iloc[0] - 0.05)) & (df['Lat'] < (flux_site_loc.loc[flux_site_loc['Site'] == col]['Y'].iloc[0] + 0.05))]
    df_gosat2 = df_gosat1.loc[(df_gosat1['Lon'] >= (flux_site_loc.loc[flux_site_loc['Site'] == col]['X'].iloc[0] - 0.05)) & (df_gosat1['Lon'] < (flux_site_loc.loc[flux_site_loc['Site'] == col]['X'].iloc[0] + 0.05))]
    #let's first check if Lat&Lon conditions satisfy
    return df_gosat2



columns = flux_site_loc['Site'].tolist()

for c in columns:
    print(func(c, df_gosat))


def func2(col, d):
    df_gosat1 = d.loc[(d['Lat'] >= (flux_site_loc.loc[flux_site_loc['Site'] == col]['Y'].iloc[0] - 0.01)) & (d['Lat'] < (flux_site_loc.loc[flux_site_loc['Site'] == col]['Y'].iloc[0] + 0.01))]
    df_gosat2 = df_gosat1.loc[(df_gosat1['Lon'] >= (flux_site_loc.loc[flux_site_loc['Site'] == col]['X'].iloc[0] - 0.01)) & (df_gosat1['Lon'] < (flux_site_loc.loc[flux_site_loc['Site'] == col]['X'].iloc[0] + 0.01))]
    #let's first check if Lat&Lon conditions satisfy
    df_co2_null = df[df['CO2'].isnull()]
    df_co2_null2 = df_co2_null[df_co2_null['Site'] == col]
    df_co2_null2['Date'] = pd.to_datetime(df_co2_null2['Date']).dt.date
    date_list = df_co2_null2['Date'].to_numpy()
    df_gosat2 = df_gosat2[df_gosat2['Date'].isin(date_list)]

    return df_gosat2



for c in columns:
    print(func2(c, df_oco))