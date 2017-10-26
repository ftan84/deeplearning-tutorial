import numpy as np
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 7
np.random.seed(seed)

dataframe = read_csv('data/sonar.csv', header=None)
dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]
Y = pd.get_dummies(Y).M.values
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)

def mlp_01():
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

def mlp_02():
    model = Sequential()
    model.add(Dense(30, input_dim=60, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

def mlp_03():
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

# mlp1
estimator = []
estimator.append(('standardize', StandardScaler()))
estimator.append(('mlp', KerasClassifier(build_fn=mlp_01, epochs=100, batch_size=5,
                                         verbose=0)))
pipeline = Pipeline(estimator)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print('Baseline: %.2f%% (%.2f%%)' % (results.mean() * 100, results.std() * 100))

# mlp2
estimator = []
estimator.append(('standardize', StandardScaler()))
estimator.append(('mlp', KerasClassifier(build_fn=mlp_02, epochs=100, batch_size=5,
                                         verbose=0)))
pipeline = Pipeline(estimator)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print('MLP2: %.2f%% (%.2f%%)' % (results.mean() * 100, results.std() * 100))

# mlp3
estimator = []
estimator.append(('standardize', StandardScaler()))
estimator.append(('mlp', KerasClassifier(build_fn=mlp_03, epochs=100, batch_size=5,
                                         verbose=0)))
pipeline = Pipeline(estimator)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print('MLP3: %.2f%% (%.2f%%)' % (results.mean() * 100, results.std() * 100))

estimator = []
estimator.append(('standardize', StandardScaler()))
estimator.append(('mlp', KerasClassifier(build_fn=mlp_03, epochs=1000,
                                         batch_size=5, verbose=1)))
pipeline = Pipeline(estimator)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print('MLP3: %.2f%% (%.2f%%)' % (results.mean() * 100, results.std() * 100))
