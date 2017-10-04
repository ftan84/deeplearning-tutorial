from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import (cross_val_score, GridSearchCV,
                                     StratifiedKFold)
import numpy as np

def create_model(optimizer='rmsprop', init='glorot_uniform'):
    """Wrapper function to create mlp model."""
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer=init,
                    activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return(model)

seed = 12080
np.random.seed(seed)

# Load dataset
dataset = np.loadtxt('9-2/data/pima.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

# Create model
model = KerasClassifier(build_fn=create_model, verbose=2)

# Build grid search
optimizers = ['rmsprop', 'adam']
inits = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches,
                  init=inits)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x, y)

# Summarize grid search
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, stdev, param))

# Use kfold cross validation
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(model, x, y, cv=kfold)
# print(results.mean())
