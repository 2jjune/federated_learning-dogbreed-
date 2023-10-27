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
conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
def create_model():
    model = models.tf.keras.Sequential()
    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(120,activation='softmax'))

    conv_base.trainable = False
    print('conv_base layer 동결 훈련 가능 가중치 종류:', len(model.trainable_weights))
    return model

model = []
epoch = 100
set_num = 10
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
new_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
avg_test_loss, avg_test_acc = new_model.evaluate(validation_generator)

new_model.set_weights(new_weights_median)
new_model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
test_loss, test_acc = new_model.evaluate(validation_generator)

print('avg 테스트 정확도,loss:', avg_test_acc, avg_test_loss)
print('median 테스트 정확도,loss:', test_acc, test_loss)