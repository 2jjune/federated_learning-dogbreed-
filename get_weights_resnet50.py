import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

df_label = pd.read_csv('dog-breed-identification/labels.csv')
label_name = df_label['breed'].sort_values().unique()
encoder = LabelEncoder()
encoder.fit(df_label['breed'])
df_label['breed'] = encoder.transform(df_label['breed'])
df_label['breed'] = df_label['breed'].astype(str)
df_label['id'] = df_label['id'] + '.jpg'

train_dir = 'dog-breed-identification/train'
batch_size=100

datagen = ImageDataGenerator(rescale=1 / 255.,
                             rotation_range = 30,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             fill_mode = 'reflect',
                             zoom_range = 0.2,
                             horizontal_flip = True,
                             vertical_flip = True,
                             validation_split=0.2)
train_generator = datagen.flow_from_dataframe(dataframe=df_label,
                                              directory=train_dir,
                                              x_col='id',
                                              y_col='breed',
                                              target_size=(224, 224),
                                              class_mode='categorical',
                                              batch_size=batch_size,
                                              shuffle = True,
                                              subset='training',
                                              seed=7)
validation_generator = datagen.flow_from_dataframe(dataframe=df_label,
                                                   directory=train_dir,
                                                   x_col='id',
                                                   y_col='breed',
                                                   target_size=(224, 224),
                                                   class_mode='categorical',
                                                   batch_size=batch_size,
                                                   shuffle = True,
                                                   subset='validation',
                                                   seed=7)

from tensorflow.keras.applications import VGG16
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import xception
from tensorflow.keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.applications import nasnet
from tensorflow.keras import Input
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten,BatchNormalization,Activation
from tensorflow.keras.layers import AveragePooling2D, ZeroPadding2D, Add
from keras.layers.merge import concatenate
from keras.initializers import glorot_uniform
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

input_shape = (224, 224, 3)
# create an Inception block

def create_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=1, strides=2))
    model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(120, activation='softmax'))

    return model

set_epoch = 100
set_num=5

model = []
for i in range(set_num):
    model.append(create_model())

for i in range(set_num):
    print(model[i])
    # model[i].compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    # model[i].compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model[i].compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

for i in range(len(model)):
    print()
    print(i,"*******************************")
    model[i].fit(train_generator, steps_per_epoch=len(train_generator)-1, epochs=set_epoch,
                              validation_data=validation_generator, validation_steps=len(validation_generator)-1)
    model[i].save_weights('cnn_{}epoch {}.h5'.format(set_epoch, i+1))
