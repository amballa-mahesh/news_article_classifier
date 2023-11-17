import pandas as pd
import numpy as np
import os
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
le = LabelEncoder()
cv = CountVectorizer(min_df=5, ngram_range=(1,4) )

from src.utils import remove_html_tags,remove_urls,word_corrections,remove_punctions_betterway,remove_stop_words,spacy_tokenisation,stemming_data,text_convert


df  = pd.read_csv('artifacts/files/BBC News Train.csv')
df1  = pd.read_csv('artifacts/files/BBC News Test.csv')
logging.info('original data read and loaded via pandas')
print('original data read and loaded via pandas')

df.drop('ArticleId',axis =1,inplace = True)
df1.drop('ArticleId',axis =1,inplace = True)

df_1 =df

for i in range(10):
  df_add = df_1.sample(500,replace=True,axis = 0)
  df_1 = pd.concat([df_1,df_add],ignore_index=True)
  
df = df_1
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

logging.info('Categories changed ot label encoding..')  
print('Categories changed ot label encoding..')  
df['Text']  = df['Text'].apply(text_convert)
df1['Text'] = df1['Text'].apply(text_convert)

df_text = pd.concat([df['Text'],df1['Text']],ignore_index=True)
df.to_csv('artifacts/data/cleaned_Train.csv')
logging.info('cleaned train data saved')
print('cleaned train data saved')
df1.to_csv('artifacts/data/cleaned_Test.csv')
logging.info('cleaned test data saved')
print('cleaned test data saved')
df_text.to_csv(('artifacts/data/cleaned_data.csv'))
logging.info('cleaned data for fit')
print('cleaned data for fit')


print(df_text.shape)
