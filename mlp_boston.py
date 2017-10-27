import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold
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

# model = KerasRegressor(build_fn=baseline, verbose=0)
pipeline = Pipeline([
    ('standardize', StandardScaler()),
    ('mlp', KerasRegressor(build_fn=baseline, verbose=2))
])
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=dict(
        mlp__epochs=[100, 250, 500],
        mlp__batch_size=[1, 5, 10]
    ),
    n_jobs=-1,
    cv=KFold(n_splits=10, shuffle=True, random_state=seed),
    verbose=2
)
grid_result = grid.fit(X.values, Y.values)

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
ckpt = ModelCheckpoint('./ckpt/model-{epoch:02d}-{val_loss:.2f}.hdf5',
                       monitor='val_loss',
                       verbose=1,
                       save_best_only=True)
callbacks = [ckpt]
history = model.fit(
    X.values,
    Y.values,
    validation_split=0.2,
    epochs=500,
    batch_size=1,
    verbose=2,
    callbacks=callbacks
)
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
