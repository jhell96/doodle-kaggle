from keras.applications.nasnet import NASNetMobile as NASNet
from keras.applications.xception import Xception
from keras.applications.mobilenetv2 import MobileNetV2


from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.client import device_lib
import os
import numpy as np
import tensorflow as tf
import cv2
from keras import backend as K
from keras.backend.tensorflow_backend import set_session


# In[2]:


K.set_learning_phase(0)

TRAIN_PATH = '../../data/img_data/small/recognized'
# TEST_PATH = '../../data/img_data/recognized'

# batch size used for validation and training
BATCH_SIZE = 32 

# Number of passes over the data
TRAIN_EPOCHS = 100

# Percent of the data used for validation
VAL_SPLIT = 0.2

# size of our images
IMG_SIZE = (256, 256)

# determines the number of classes and total # of samples
walk_dir = os.walk(TRAIN_PATH)

# NUM_CLASSES = len(list(walk_dir)[0][1])
NUM_CLASSES = 340
NUM_SAMPLES = 45000
# for root, dirs, files in walk_dir:
#     NUM_SAMPLES += len(files)


# In[3]:


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


# In[4]:


single_threaded_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), weights=None, classes=NUM_CLASSES)
model = multi_gpu_model(single_threaded_model, gpus=4)
model.compile(optimizer=Adam(lr=0.02), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
print(model.summary())


# In[5]:


# make transformers
train_datagen = ImageDataGenerator(
#         rotation_range=30,
#         shear_range=0.2,
#         zoom_range=[0.9, 1],
#         horizontal_flip=True,
#         fill_mode='nearest',
        validation_split=VAL_SPLIT)


# In[6]:


train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        interpolation='nearest',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        interpolation='nearest',
        subset='validation')


# In[7]:


# define callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5,
                      min_delta=0.005, mode='max', cooldown=3, verbose=1)
]


# In[8]:


history = model.fit_generator(
            train_generator,
            steps_per_epoch=((1-VAL_SPLIT)*NUM_SAMPLES)//BATCH_SIZE,
            epochs=TRAIN_EPOCHS,
            validation_data=validation_generator,
            validation_steps=(VAL_SPLIT*NUM_SAMPLES)//BATCH_SIZE)

