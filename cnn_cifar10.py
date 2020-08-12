# 3 conv layers- 100 epochs: accuracy 0.7090 val_accuracy 0.7486

################################
########### Image Preprocessing

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=32)


################################
########### Building CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout  ## To add Dropout Regularization

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3), padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dropout(0.5))
classifier.add(Dense(units = 128, activation ='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 10, activation ='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit_generator(
        train_generator,
        steps_per_epoch=len(x_train),
        epochs=100,
        validation_data=validation_generator,
        validation_steps=len(x_test))

#classifier.summary() 
classifier.save('model_cifar10.h5')
