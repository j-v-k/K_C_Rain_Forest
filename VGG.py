import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import time

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import cv2
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
from tqdm import tqdm

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
import time
print("running")

os.chdir("C:\\Users\\James\\Documents\\GitHub\\NeuralNetwork_comp")
print("running")
x_train = []
x_test = []
y_train = []

df_train = pd.read_csv('input/train.csv')
df_test = pd.read_csv('input/sample_submission.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))
#df_train = None
labels = ['blow_down',
 'bare_ground',
 'conventional_mine',
 'blooming',
 'cultivation',
 'artisinal_mine',
 'haze',
 'primary',
 'slash_burn',
 'habitation',
 'clear',
 'road',
 'selective_logging',
 'partly_cloudy',
 'agriculture',
 'water',
 'cloudy']

label_map = {'agriculture': 14,
 'artisinal_mine': 5,
 'bare_ground': 1,
 'blooming': 3,
 'blow_down': 0,
 'clear': 10,
 'cloudy': 16,
 'conventional_mine': 2,
 'cultivation': 4,
 'habitation': 9,
 'haze': 6,
 'partly_cloudy': 13,
 'primary': 7,
 'road': 11,
 'selective_logging': 12,
 'slash_burn': 8,
 'water': 15}

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('input/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (224, 224)))
    y_train.append(targets)

for f, tags in tqdm(df_test.values, miniters=1000):
    img = cv2.imread('input/test-jpg/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (224, 224)))

y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32) / 255.
x_test = np.array(x_test, np.float32) / 255.
import pickle

import pickle
pickle.dump( x_test, open( "x_test_224.p", "wb" ) )
pickle.dump( x_train, open( "x_train_224.p", "wb" ) )
pickle.dump( y_train, open( "y_train_224.p", "wb" ) )
print(x_train.shape)
print(y_train.shape)


# https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
import numpy as np
from sklearn.metrics import fbeta_score

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*17
  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= resolution
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x




nfolds = 2

num_fold = 0
sum_score = 0

yfull_test = []
yfull_train = []
split = 3000
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float32) / 255
# kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1)
X_train, X_valid, Y_train, Y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]
print(len(X_valid))

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH = 'C:\\Users\\James\\Documents\\GitHub\\NeuralNetwork_comp\\vgg16_weights_th_dim_ordering_th_kernels.h5'
# Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()
vgg16_weights_path = "vgg16_weights_th_dim_ordering_th_kernels.h5"
# model_vgg16_conv.load_weights(vgg16_weights_path)
# Create your own input format (here 3x200x200)
input = Input(shape=(224, 224, 3), name='image_input')

# Use the generated model
output_vgg16_conv = model_vgg16_conv(input)

# Add the fully-connected layers
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(17, activation='softmax', name='predictions')(x)

# Create your own model
model = Model(input=input, output=x)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
          batch_size=164, verbose=2, nb_epoch=2,
          shuffle=True)

p_valid = model.predict(X_valid, batch_size=164, verbose=2)
print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
# print("Optimizing prediction threshold")
# print(optimise_f2_thresholds(Y_valid, p_valid))

# if os.path.isfile(kfold_weights_path):
# model.load_weights(kfold_weights_path)

p_test = model.predict(x_train, batch_size=128, verbose=2)
yfull_train.append(p_test)

p_test = model.predict(x_test, batch_size=128, verbose=2)
yfull_test.append(p_test)

result = np.array(yfull_test[0])
for i in range(1, nfolds):
    result += np.array(yfull_test[i])
result /= nfolds
result = pd.DataFrame(result, columns = labels)
print(len(result))
#thres =
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    #a = a.apply(lambda x: x > thres, axis=1)
    a = a.apply(lambda x: x > .2, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test['tags'] = preds