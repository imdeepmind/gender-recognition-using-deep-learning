import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

data = pd.read_csv('faces.csv')

X_data = data.drop(['class'], axis=1)
y_data = data['class']

X = X_data.values.reshape(-1,32,32,1)
y = pd.get_dummies(y_data).values

X = X / 255.0

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(264, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X, y, validation_split = 0.2, epochs=25, batch_size=64)

# loss: 0.2859 - acc: 0.8833 - val_loss: 0.4149 - val_acc: 0.8025
# loss: 0.2754 - acc: 0.8821 - val_loss: 0.3682 - val_acc: 0.8380
# loss: 0.3097 - acc: 0.8706 - val_loss: 0.3533 - val_acc: 0.8354
# loss: 0.0287 - acc: 0.9924 - val_loss: 0.8390 - val_acc: 0.8177
# loss: 0.0905 - acc: 0.9645 - val_loss: 0.7569 - val_acc: 0.8253
# loss: 0.2527 - acc: 0.8871 - val_loss: 0.4390 - val_acc: 0.7949