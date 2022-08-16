from sklearn.metrics import mean_squared_error,r2_score
import pickle
from scripts.data_pipeline import X_test_scaled, y_test
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from scripts.data_pipeline import scaler, feature_engineering1, feature_engineering2, onehot_encoder, onehotcols
import folium
from folium.features import DivIcon


app = Flask(__name__)

#Loading the model 

def load_model(PATH):
    with open(PATH, 'rb') as f:
        final_model = pickle.load(f)
    return final_model

model = load_model('./models/model.pkl')


#Testing the loaded model:

def test():
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return f'R2 is {round(r2,2)}', f'RMSE is {round(rmse,2)}'
    

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print(request.files['file'])
        f = request.files['file']
        df2 = pd.read_excel(f)
        df2_copy = df2.copy()
        df2_copy = feature_engineering1(df2_copy)
        df2_copy = feature_engineering2(df2_copy)
        df2_copy = onehot_encoder(df2_copy,onehotcols)
        df2_copy = df2_copy.drop(['Date'],axis=1)


        #Temporary solution for the sklearn bug:

        df2_copy.rename(columns = {'month_1':'month_1.0', 
                                    'month_2':'month_2.0',
                                    'month_3':'month_3.0',
                                    'month_4':'month_4.0',
                                    'month_5':'month_5.0',
                                    'month_6':'month_6.0',
                                    'month_7':'month_7.0',
                                    'month_8':'month_8.0',
                                    'month_9':'month_9.0',
                                    'month_10':'month_10.0',
                                    'month_11':'month_11.0',
                                    'month_12':'month_12.0',
                                    }, inplace = True)


        df2_copy_scaled = scaler.transform(df2_copy)

        #Getting predictions

        predictions = model.predict(df2_copy_scaled)
        df2['NEE'] = predictions.tolist()

        #Rounding floats
        r_cols = df2.select_dtypes(include=[np.number])
        df2.loc[:, r_cols.columns] = np.round(r_cols,2)

        pd.set_option('colheader_justify', 'center')

        global df5
        df5 = df2

        HEADER = '''
        <html>
            <head>
                <link rel="stylesheet" href="./static/css/style.css">
            </head>
            <body>
            <ul class="ul_links">
                <a href="{{ url_for('index') }}">Home page</a>
                <a href="{{ url_for('display_map') }}" target="_blank">Map</a>
            </ul>
        '''
        FOOTER = '''
            </body>
        </html>
        '''

        with open('./templates/test.html', 'w') as f:
            f.write(HEADER)
            f.write(df2.to_html(index=False, col_space=80, classes='df2'))
            f.write(FOOTER)


        return render_template('test.html')


@app.route('/display_map', methods=['GET', 'POST'])
def display_map():
    df_latlon = pd.read_excel('./data/latlon.xls') 
    df3 = df5.merge(df_latlon, how='left', on='Site')
    df_lon = df3.groupby('Site', as_index=False)['Longitude'].mean()
    df_lat = df3.groupby('Site', as_index=False)['Latitude'].mean()
    df_lat_lon = df_lat.merge(df_lon, how='left', on='Site')
    df3 = df3.groupby('Site', as_index=False)['NEE'].mean()
    df3 = df3.merge(df_lat_lon, how='left', on='Site')
    df3['latlon'] = list(zip(df3.Latitude, df3.Longitude))


    nee_map = folium.Map(
        location=[30, -80],
        zoom_start=5,
    )

        
    for i in df3['latlon'].loc[:]:
        index = df3.loc[df3['latlon'] == i].index[0]
        site = df3['Site'].iloc[index]
        nee_value = df3['NEE'].iloc[index]
        nee_value = np.round(nee_value,2)
        folium.Marker(i, icon=DivIcon(
            icon_size=(10,10),
            icon_anchor=(20,20),
            html=f'<div style="font-size: 10pt; font-weight: bold; color : black">{site}: {nee_value}</div>',
            )).add_to(nee_map)
        
        if nee_value>0:
            nee_map.add_child(folium.CircleMarker(i, fill_color="green", color = 'green', radius=40))
        else:
            nee_map.add_child(folium.CircleMarker(i, fill_color="red", color = 'red', radius=40))

        
    nee_map.save('./templates/map.html')

    return render_template('map.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)



