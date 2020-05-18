#!/usr/bin/python36
from keras.layers import Convolution2D, MaxPooling2D ,Flatten, Dense
from keras.models import Sequential
model=Sequential()

model.add(Convolution2D(filters=32,kernel_size=(2,2),activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D())
#model.summary()

model.add(Convolution2D(filters=32,kernel_size=(2,2),activation='relu'))
model.add(MaxPooling2D())
#model.summary()

model.add(Flatten())
#model.summary()

model.add(Dense(units=128,activation='relu'))
#model.summary()

model.add(Dense(units=1,activation='sigmoid'))
#model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#model.summary()

from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'images/train/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
val_set = train_datagen.flow_from_directory('images/val/', 
                                      class_mode='binary',
                                      target_size=(64, 64),
                                       batch_size=32)

test_set = test_datagen.flow_from_directory(
        'images/test/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
history=model.fit(
        training_set,
        steps_per_epoch=10,
        epochs=10,
        validation_data=val_set,
        validation_steps=5)



final_accuracy=history.history['accuracy'][-1]
print(final_accuracy)
model.save("room_clean_messy.h5")





