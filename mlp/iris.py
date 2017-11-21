import numpy as np
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from dplython import *
# from dplython import (DplyFrame, X, diamonds, select, sift, head, arrange,
#                       mutate, group_by, summarize)
seed = 12080
np.random.seed(seed)

dataset = DplyFrame(read_csv('data/iris.csv', header=None))
x = (dataset >>
        select(X[0], X[1], X[2], X[3]))
y = (dataset >>
        select(X[4]))
y = pd.get_dummies(y)

def baseline_model():
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return(model)

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5,
                            verbose=2)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, x.values, y.values, cv=kfold)
results.mean() * 100
