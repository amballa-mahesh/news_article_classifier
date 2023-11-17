import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from pickle import dump
from pickle import load
from src.utils import tokenized_text, max_len_captions, create_seq
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout,Dense, LSTM,Bidirectional
from src.utils import text_pipeline,model_predict
from keras.models import load_model
import joblib
import os

tokenizer = load(open("artifacts/data/tokenizer.p","rb"))
print('tokenizer loaded...')

df = pd.read_csv('artifacts/data/cleaned_Train.csv')


token_text = tokenized_text(df['Text'],tokenizer)
max_len = max_len_captions(df['Text'])
print(max_len)


X = create_seq(token_text,max_len)
y = to_categorical(df['Category'])
y_le = df['Category']


x_train,x_test,y_train,y_test = train_test_split(X,y_le,test_size = 0.1,random_state = 42)


#Random Forest

model_rf = RandomForestClassifier()
model_rf.fit(x_train,y_train)
y_pred = model_rf.predict(x_test)
score_rf = np.round(accuracy_score(y_test,y_pred),2)
print(score_rf)
print(classification_report(y_test,y_pred))

joblib.dump(model_rf,os.path.join('artifacts','models','model_rf.pkl'))

# Support vector machine
# model_svc = SVC()
# model_svc.fit(x_train,y_train)
# y_pred = model_svc.predict(x_test)
# score_svc = np.round(accuracy_score(y_test,y_pred),2)
# print(score_svc)
# print(classification_report(y_test,y_pred))


# neural network model
# x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 42)

# model_nn = Sequential()
# model_nn.add(Dense(1632,activation = 'relu',input_shape = [1632,]))
# model_nn.add(Dropout(rate =0.2))
# model_nn.add(Dense(5,activation = 'softmax'))

# model_nn.compile(optimizer= 'adam',loss='categorical_crossentropy',metrics ='accuracy')

# model_nn.fit(np.array(x_train),y_train,epochs =10,verbose=2,batch_size=100,validation_split=0.1)

# y_pred = np.round(model_nn.predict(np.array(x_test)),0)
# score_nn = np.round(accuracy_score(y_test,y_pred),2)

# print(print('rf-score {},svc_score {},NN_score {}'.format(score_rf,score_svc,score_nn)))

# model_nn.save("artifacts/models/model_nn.h5")

