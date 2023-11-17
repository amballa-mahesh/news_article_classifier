import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from pickle import load
from src.utils import text_pipeline,model_predict_nn,model_predict_rf
from keras.models import load_model
import joblib
from joblib import dump, load


tokenizer = load(open("artifacts/data/tokenizer.p","rb"))
print('tokenizer loaded...')

# model_nn = load_model('artifacts/models/model_rf.pkl')
# print('Model loaded...')

model_rf = joblib.load('artifacts/models/model_rf.pkl')

while True:
    txt = input('Enter your article here to classify-- ')
    X =  text_pipeline(txt,tokenizer)
    result = model_predict_rf(X,model_rf)


