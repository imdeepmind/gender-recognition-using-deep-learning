import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

data = pd.read_csv('dataset/faces.csv')

X_data = data.drop(['class'], axis=1)
y_data = data['class']

X = X_data.values.reshape(-1,64,64,1)
y = pd.get_dummies(y_data).values

X = X / 255.0

# model = Sequential()

# model.add(Conv2D(input_shape = (64,64,1), filters=32, kernel_size=(5,5), padding='same', activation='relu'))
# model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
# model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(2, activation='softmax'))

model = Sequential()
#Layer 1
#Conv Layer 1
model.add(Conv2D(filters = 6, 
                 kernel_size = 5, 
                 strides = 1, 
                 activation = 'relu', 
                 input_shape = (64,64,1)))
#Pooling layer 1
model.add(MaxPool2D(pool_size = 2, strides = 2))
#Layer 2
#Conv Layer 2
model.add(Conv2D(filters = 32, 
                 kernel_size = 5,
                 strides = 1,
                 activation = 'relu'))


#Pooling Layer 2
model.add(MaxPool2D(pool_size = 2, strides = 2))

# model.add(Dropout(0.25))

#Flatten
model.add(Flatten())
#Layer 3
#Fully connected layer 1
model.add(Dense(units = 512, activation = 'relu'))
#Layer 4
#Fully connected layer 2
model.add(Dense(units = 128, activation = 'relu'))

model.add(Dense(units = 64, activation = 'relu'))

model.add(Dense(units = 28, activation = 'relu'))

model.add(Dense(units = 16, activation = 'relu'))

model.add(Dense(units = 8, activation = 'relu'))

model.add(Dense(units = 4, activation = 'relu'))
model.add(Dropout(0.2))
#Layer 5
#Output Layer
model.add(Dense(units = 2, activation = 'softmax'))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=2)

datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,  
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1,
        horizontal_flip=True,  
        vertical_flip=False)  

datagen.fit(X_train)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data = (X_test,y_test),
                    steps_per_epoch=len(X_train) / 32, epochs=20)

hist_df = pd.DataFrame(history.history)

fig = plt.figure(figsize=(14,6))
plt.style.use('bmh')
params_dict = dict(linestyle='solid', linewidth=0.25, marker='o', markersize=6)

plt.subplot(121)
plt.plot(hist_df.loss, label='Training loss', **params_dict)
plt.plot(hist_df.val_loss, label='Validation loss', **params_dict)
plt.title('Loss for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


# with this model im getting about 78% to 81% accuracy
# need to optimize the model and the dataset to acheive better result (+98%)

plt.subplot(122)
plt.plot(hist_df.acc, label='Training accuracy', **params_dict)
plt.plot(hist_df.val_acc, label='Validation accuracy', **params_dict)
plt.title('Accuracy for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()