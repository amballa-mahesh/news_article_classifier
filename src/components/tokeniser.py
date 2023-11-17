import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from pickle import dump
from pickle import load

df_text = pd.read_csv('artifacts/data/cleaned_data.csv')

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df_text['Text'])

dump(tokenizer, open('artifacts/data/tokenizer.p', 'wb'))
