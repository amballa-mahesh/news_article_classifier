import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from pickle import load
from src.utils import text_pipeline,model_predict,text_preprocess
from keras.models import load_model


tokenizer = load(open("artifacts/data/tokenizer.p","rb"))
print('tokenizer loaded...')

data = text_pipeline('HEllo hoW are You TODAY 123 a . ',tokenizer)