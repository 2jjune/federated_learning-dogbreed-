import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import Input
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten,BatchNormalization,Activation
from tensorflow.keras.layers import AveragePooling2D, ZeroPadding2D
from keras.layers.merge import concatenate

df_label = pd.read_csv('dog-breed-identification/labels.csv')
label_name = df_label['breed'].sort_values().unique()
encoder = LabelEncoder()
encoder.fit(df_label['breed'])
df_label['breed'] = encoder.transform(df_label['breed'])
df_label['breed'] = df_label['breed'].astype(str)
df_label['id'] = df_label['id'] + '.jpg'

train_dir = 'dog-breed-identification/train'
test_dir = 'dog-breed-identification/test'
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
input_shape = (224, 224, 3)
# create an Inception block

def inception(x, filters):
    # 1x1
    path1 = Conv2D(filters=filters[0], kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)

    # 1x1->3x3
    path2 = Conv2D(filters=filters[1][0], kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    path2 = Conv2D(filters=filters[1][1], kernel_size=(3, 3), strides=1, padding='same', activation='relu')(path2)

    # 1x1->5x5
    path3 = Conv2D(filters=filters[2][0], kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    path3 = Conv2D(filters=filters[2][1], kernel_size=(5, 5), strides=1, padding='same', activation='relu')(path3)

    # 3x3->1x1
    path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    path4 = Conv2D(filters=filters[3], kernel_size=(1, 1), strides=1, padding='same', activation='relu')(path4)

    return Concatenate(axis=-1)([path1, path2, path3, path4])

def auxiliary(x, name=None):
  layer = AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)
  layer = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
  layer = Flatten()(layer)
  layer = Dense(units=256, activation='relu')(layer)
  layer = Dropout(0.4)(layer)
  layer = Dense(units=120, activation='softmax', name=name)(layer)

  return layer

def create_model():
    layer_in = Input(shape=input_shape)

    # stage-1
    layer = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', activation='relu')(layer_in)
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
    layer = BatchNormalization()(layer)

    # stage-2
    layer = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(layer)
    layer = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)

    # stage-3
    layer = inception(layer, [64, (96, 128), (16, 32), 32])  # 3a
    layer = inception(layer, [128, (128, 192), (32, 96), 64])  # 3b
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)

    # stage-4
    layer = inception(layer, [192, (96, 208), (16, 48), 64])  # 4a
    aux1 = auxiliary(layer, name='aux1')
    layer = inception(layer, [160, (112, 224), (24, 64), 64])  # 4b
    layer = inception(layer, [128, (128, 256), (24, 64), 64])  # 4c
    layer = inception(layer, [112, (144, 288), (32, 64), 64])  # 4d
    aux2 = auxiliary(layer, name='aux2')
    layer = inception(layer, [256, (160, 320), (32, 128), 128])  # 4e
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)

    # stage-5
    layer = inception(layer, [256, (160, 320), (32, 128), 128])  # 5a
    layer = inception(layer, [384, (192, 384), (48, 128), 128])  # 5b
    layer = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')(layer)

    # stage-6
    layer = Flatten()(layer)
    layer = Dropout(0.4)(layer)
    layer = Dense(units=256, activation='linear')(layer)
    main = Dense(units=120, activation='softmax', name='main')(layer)

    model = tf.keras.Model(inputs=layer_in, outputs=[main, aux1, aux2])

    return model

model = []
epoch = 50
set_num = 5
for i in range(set_num):
    model.append(create_model())

new_model = create_model()

for i in range(set_num):
    model[i].load_weights("{}epoch {}.h5".format(epoch, i+1))

weights = []
for i in range(set_num):
    weights.append(model[i].get_weights())
print('weights 갯수 : ',len(weights))


new_weights_avg = list()
new_weights_median = list()
for weights_list_tuple in zip(*weights):
    new_weights_avg.append(
        np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        # np.array([np.median(np.array(w), axis=0) for w in zip(*weights_list_tuple)])
    )
    new_weights_median.append(
        np.array([np.median(np.array(w), axis=0) for w in zip(*weights_list_tuple)])
    )

#모델 evaluate
new_model.set_weights(new_weights_avg)
new_model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
avg_loss, avg_main_loss, avg_aux1_loss, avg_aux2_loss, avg_main_acc, avg_aux1_acc, avg_aux2_acc = new_model.evaluate(validation_generator)

new_model.set_weights(new_weights_median)
new_model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
median_loss, median_main_loss, median_aux1_loss, median_aux2_loss, median_main_acc, median_aux1_acc, median_aux2_acc = new_model.evaluate(validation_generator)

print('avg 테스트 loss:{}, main_loss:{}, aux1_loss:{}, aux2_loss:{}'.format(avg_loss, avg_main_loss, avg_aux1_loss, avg_aux2_loss))
print('avg 테스트 main_acc:{}, aux1_acc:{}, aux2_acc:{}'.format(avg_main_acc, avg_aux1_acc, avg_aux2_acc))
print('median 테스트 loss:{}, main_loss:{}, aux1_loss:{}, aux2_loss:{}'.format(median_loss, median_main_loss, median_aux1_loss, median_aux2_loss))
print('median 테스트 main_acc:{}, aux1_acc:{}, aux2_acc:{}'.format(median_main_acc, median_aux1_acc, median_aux2_acc))
