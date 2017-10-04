from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

seed = 12080
np.random.seed(seed)

# Load data
dataset = np.loadtxt('7-1/data/pima.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=seed)

# Create the network
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=1000, batch_size=10, validation_split=0.2, verbose=2)

# Evaluate model
scores = model.evaluate(x_test, y_test, verbose=0)
print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
