import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

data = pd.read_csv('../dataset/faces.csv')

X_data = data.drop(['class'], axis=1)
y_data = data['class']

X = X_data.values.reshape(-1,32,32,1)
y = pd.get_dummies(y_data).values

X = X / 255.0

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X, y, validation_split = 0.2, epochs=10, batch_size=64)

# with this model - loss: 0.2859 - acc: 0.8833 - val_loss: 0.4149 - val_acc: 0.8025

