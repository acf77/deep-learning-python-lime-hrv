from statistics import mode
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# import lime
# from lime import lime_tabular

# load the dataset
dataset = loadtxt('holter-cad-dia-dt_df_tr_vic.txt', delimiter=';')
# split into input (X) and output (y) variables
X = dataset[:, 0:25]
y = dataset[:, 25]

# print(dataset)


# define the keras model
model = Sequential()
model.add(Dense(256, input_dim=25, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
# model.add(Dropout(0.3))
# model.add(Dense(2, activation='relu'))


# compile the keras model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=100, batch_size=5)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

# LIME explanation
# explainer = lime_tabular.LimeTabularExplainer(X, mode='classification')
