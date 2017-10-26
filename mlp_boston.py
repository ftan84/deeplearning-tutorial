import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def baseline():
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

seed = 12080
np.random.seed(seed)

df = pd.read_csv('data/housing.data', delim_whitespace=True, header=None)
X = df.iloc[:, 0:13]
Y = df.iloc[:, 13]

# regressor = KerasRegressor(build_fn=baseline,
#                            epochs=100,
#                            batch_size=1,
#                            verbose=0)
# pipeline = Pipeline([
#     ('standardize', StandardScaler()),
#     ('mlp', regressor)
# ])
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(
#     pipeline,
#     X=X.values,
#     y=Y.values,
#     cv=kfold,
#     n_jobs=-1,
#     verbose=2
# )
# print('Baseline: %.2f (%.2f) MSE' % (results.mean(), results.std()))

model = baseline()
history = model.fit(
    X.values,
    Y.values,
    validation_split=0.2,
    epochs=500,
    batch_size=1,
    verbose=2
)
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
