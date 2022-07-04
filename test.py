import tensorflow as tf
import numpy as np
import pandas as pd

from utils import config as cfg
from data_helper import DataLoader
from modules import models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


data_test = pd.read_csv(cfg.test_path, encoding='latin-1')
# get Train data
df_train = DataLoader(data_test, 128, 'distilroberta-base', 'clinc150')
input_test, label_test = df_train.convert_to_tensor()

# create model
eval_model = models.intent_model(cfg.numclasses)
eval_model.load_weights(cfg.intent_path)

# get embds
y_test= np.zeros(len(label_test))
tf.convert_to_tensor(y_test)
_, embds = eval_model.predict(x=(input_test, y_test))

# Create KNN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(embds, label_test)
A = neigh.predict(embds)

# Report classification
label = label_test.numpy().reshape(-1,)
print(classification_report(label, A))