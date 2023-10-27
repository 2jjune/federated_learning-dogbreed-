import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tensorflow import keras

import cv2
import PIL
import os
import pathlib
import shutil
from IPython.display import Image, display

# Plotly for the interactive viewer (see last section)

import plotly.graph_objs as go
import plotly.graph_objects as go
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import xception
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.applications import nasnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten,BatchNormalization,Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing

import gc
import skimage.io


train_dir = 'dog-breed-dataset-with-subdirectories-by-class/data/train'
test_dir = 'dog-breed-dataset-with-subdirectories-by-class/data/test'

train_labels = pd.read_csv('dog-breed-identification/labels.csv', index_col = 'id')
submission=pd.read_csv('dog-breed-identification/sample_submission.csv')

train_size = len(os.listdir(train_dir))
test_size = len(os.listdir(test_dir))

print(train_size,test_size)
print(train_labels.shape)
print(submission.shape)

target, dog_breeds = pd.factorize(train_labels['breed'], sort = True)
train_labels['target'] = target
train_labels['breed'].value_counts()


BATCH_SIZE = 128
IMG_HEIGHT = 331
IMG_WIDTH = 331

train_ds = image_dataset_from_directory(
  directory = train_dir,
  labels = 'inferred',
  label_mode='int',
  batch_size=BATCH_SIZE,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  shuffle = True,
  seed=1234,
  validation_split=0.1,
  subset="training",
)

class_names = train_ds.class_names

val_ds = image_dataset_from_directory(
  directory = train_dir,
  labels = 'inferred',
  label_mode='int',
  batch_size=BATCH_SIZE,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  shuffle = True,
  seed=1234,
  validation_split=0.1,
  subset="validation",
)

AUTOTUNE = tf.data.AUTOTUNE

# Without Caching
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


data_augmentation = Sequential(
  [
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.1),
    preprocessing.RandomZoom(0.1),
  ]
)

def create_model():
  base_model_1 = xception.Xception(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
  base_model_2 = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
  base_model_3 = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
  # base_model_4 = resnet_v2.ResNet152V2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH,3))
  base_model_5 = nasnet.NASNetLarge(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
  base_model_6 = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
  base_model_7 = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

  # display(base_model_1.summary())
  # display(base_model_2.summary())
  # display(base_model_3.summary())
  # display(base_model_4.summary())
  # display(base_model_5.summary())
  # display(base_model_6.summary())
  # display(base_model_7.summary())

  # train only the top layers (which were randomly initialized)
  # i.e. freeze all convolutional Xception layers
  base_model_1.trainable = False
  base_model_2.trainable = False
  base_model_3.trainable = False
  # base_model_4.trainable = False
  base_model_5.trainable = False
  base_model_6.trainable = False
  base_model_7.trainable = False

  inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
  aug_inputs = data_augmentation(inputs)

  ## <-----  Xception   -----> ##
  x1 = xception.preprocess_input(aug_inputs)
  x1 = base_model_1(x1, training=False)
  x1 = GlobalAveragePooling2D()(x1)

  ## <-----  InceptionV3   -----> ##
  # x2 = inception_v3.preprocess_input(aug_inputs)
  # x2 = base_model_2(x2, training=False)
  # x2 = GlobalAveragePooling2D()(x2)

  ## <-----  InceptionResNetV2   -----> ##
  # x3 = inception_resnet_v2.preprocess_input(aug_inputs)
  # x3 = base_model_3(x3, training=False)
  # x3 = GlobalAveragePooling2D()(x3)

  ## <-----  ResNet152V2   -----> ##
  # x4 = resnet_v2.preprocess_input(aug_inputs)
  # x4 = base_model_4(x4, training=False)
  # x4 = GlobalAveragePooling2D()(x4)

  ## <-----  NASNetLarge   -----> ##
  # x5 = nasnet.preprocess_input(aug_inputs)
  # x5 = base_model_5(x5, training=False)
  # x5 = GlobalAveragePooling2D()(x5)

  x6 = tf.keras.applications.vgg19.preprocess_input(aug_inputs)
  x6 = base_model_2(x6, training=False)
  x6 = GlobalAveragePooling2D()(x6)

  x7 = vgg16.preprocess_input(aug_inputs)
  x7 = base_model_3(x7, training=False)
  x7 = GlobalAveragePooling2D()(x7)

  ## <-----  Concatenation  -----> ##
  x = Concatenate()([x6,x7])
  x = Dropout(.7)(x)
  outputs = Dense(120, activation='softmax')(x)
  model = Model(inputs, outputs)

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
    model[i].compile(loss="sparse_categorical_crossentropy", metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))

for i in range(len(model)):
    print()
    print(i,"*******************************")
    EarlyStop_callback = EarlyStopping(min_delta=0.001, patience=10, restore_best_weights=True)
    model[i].fit(train_ds, epochs=set_epoch, validation_data=val_ds, callbacks=[EarlyStop_callback])
    model[i].save_weights('4backbone_{}epoch {}.h5'.format(set_epoch, i+1))
