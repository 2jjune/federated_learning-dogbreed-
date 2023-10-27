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
                                              target_size=(150, 150),
                                              class_mode='categorical',
                                              batch_size=batch_size,
                                              subset='training',
                                              seed=7)
validation_generator = datagen.flow_from_dataframe(dataframe=df_label,
                                                   directory=train_dir,
                                                   x_col='id',
                                                   y_col='breed',
                                                   target_size=(150, 150),
                                                   class_mode='categorical',
                                                   batch_size=batch_size,
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

data_augmentation = Sequential(
  [
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.1),
    preprocessing.RandomZoom(0.1),
  ]
)

def create_model():
    # conv_base = VGG16(weights='imagenet',
    #                   include_top=False,
    #                   input_shape=(150, 150, 3))
    #
    # model = models.tf.keras.Sequential()
    # model.add(conv_base)
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(256,activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(120,activation='softmax'))
    #
    # conv_base.trainable = False
    # print('conv_base layer 동결 훈련 가능 가중치 종류:', len(model.trainable_weights))
    IMG_HEIGHT=150
    IMG_WIDTH=150
    base_model_1 = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model_2 = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model_3 = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model_4 = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    base_model_1.trainable = False
    base_model_2.trainable = False
    base_model_3.trainable = False
    base_model_4.trainable = False

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    aug_inputs = data_augmentation(inputs)

    ## <-----  InceptionV3   -----> ##
    # x1 = inception_v3.preprocess_input(aug_inputs)
    # x1 = base_model_1(x1, training=False)
    # x1 = GlobalAveragePooling2D()(x1)

    x2 = tf.keras.applications.vgg19.preprocess_input(aug_inputs)
    x2 = base_model_2(x2, training=False)
    x2 = GlobalAveragePooling2D()(x2)

    x3 = vgg16.preprocess_input(aug_inputs)
    x3 = base_model_3(x3, training=False)
    x3 = GlobalAveragePooling2D()(x3)

    # x4 = resnet50.preprocess_input(aug_inputs)
    # x4 = base_model_4(x4, training=False)
    # x4 = GlobalAveragePooling2D()(x4)

    ## <-----  Concatenation  -----> ##
    x = Concatenate()([x2, x3])
    x = Dropout(.7)(x)
    outputs = Dense(256, activation='relu')(x)
    outputs = Dropout(.5)(outputs)
    outputs = Dense(120, activation='softmax')(outputs)
    model = Model(inputs, outputs)

    return model

set_epoch = 100
set_num=5

model = []
for i in range(set_num):
    model.append(create_model())

for i in range(set_num):
    # model[i].compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model[i].compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

for i in range(len(model)):
    print()
    print(i,"*******************************")
    model[i].fit(train_generator, steps_per_epoch=len(train_generator)-1, epochs=set_epoch,
                              validation_data=validation_generator, validation_steps=len(validation_generator)-1)
    model[i].save_weights('{}epoch {}.h5'.format(set_epoch, i+1))
