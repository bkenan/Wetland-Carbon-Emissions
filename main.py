from sklearn.metrics import mean_squared_error,r2_score
import pickle
from scripts.data_pipeline import X_test_scaled, y_test
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from scripts.data_pipeline import scaler, feature_engineering1, feature_engineering2, onehot_encoder, onehotcols



app = Flask(__name__)

#Loading the model 

def load_model(PATH):
    with open(PATH, 'rb') as f:
        final_model = pickle.load(f)
    return final_model

model = load_model('./models/model.pkl')






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)



