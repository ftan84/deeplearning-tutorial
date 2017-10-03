from keras.layers import Dense
from keras.models import Sequential
import numpy as np

np.random.seed(12080)

# Load data
dataset = np.loadtxt('pima.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

# Create the network
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(x, y, epochs=500, batch_size=10, verbose=2)

# Evaluate model
scores = model.evaluate(x, y)
print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
